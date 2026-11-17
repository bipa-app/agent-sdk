//! Frames of the AWS event-stream encoding (`application/vnd.amazon.eventstream`)
//! that `InvokeModelWithResponseStream` answers with, parsed by AWS's
//! `aws-smithy-eventstream`.
//!
//! Bytes arrive in arbitrary chunks. The decoder buffers them, reads each
//! frame's total length from its first four bytes, and hands a complete frame
//! to `read_message_from`, which checks the prelude and message CRCs and
//! parses the headers. A length outside what Bedrock sends fails before the
//! body is awaited, which bounds the buffer. A failed frame ends the stream:
//! the next frame boundary is unknown.

use std::fmt;

use aws_smithy_eventstream::frame::read_message_from;
use aws_smithy_types::event_stream::Message;
use bytes::BytesMut;

/// Smallest frame: a 12-byte prelude and a message CRC.
const MIN_MESSAGE_LEN: usize = 16;
/// Largest frame accepted. A Bedrock `chunk` carries at most 1,000,000
/// base64 characters, so 4 MiB bounds the buffer well above any real frame.
const MAX_MESSAGE_LEN: usize = 4 * 1024 * 1024;

#[derive(Debug)]
pub enum DecodeError {
    Length(usize),
    Frame(aws_smithy_eventstream::error::Error),
}

impl fmt::Display for DecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Length(length) => write!(
                formatter,
                "event-stream frame of {length} bytes, outside {MIN_MESSAGE_LEN}..={MAX_MESSAGE_LEN}"
            ),
            Self::Frame(error) => write!(formatter, "event-stream frame: {error}"),
        }
    }
}

impl std::error::Error for DecodeError {}

#[derive(Debug, Default)]
pub struct Decoder {
    buffer: BytesMut,
}

impl Decoder {
    pub fn push(&mut self, bytes: &[u8]) {
        self.buffer.extend_from_slice(bytes);
    }

    /// Bytes buffered that do not yet form a whole frame.
    pub fn pending(&self) -> usize {
        self.buffer.len()
    }

    /// The next whole message, `Ok(None)` while one is still arriving.
    pub fn next_message(&mut self) -> Result<Option<Message>, DecodeError> {
        let Some(prefix) = self.buffer.first_chunk::<4>() else {
            return Ok(None);
        };
        let length = usize::try_from(u32::from_be_bytes(*prefix)).unwrap_or(usize::MAX);
        if !(MIN_MESSAGE_LEN..=MAX_MESSAGE_LEN).contains(&length) {
            return Err(DecodeError::Length(length));
        }
        if self.buffer.len() < length {
            return Ok(None);
        }
        let frame = self.buffer.split_to(length).freeze();
        read_message_from(frame)
            .map(Some)
            .map_err(DecodeError::Frame)
    }
}

/// A string-valued header of `message`.
pub fn header<'a>(message: &'a Message, name: &str) -> Option<&'a str> {
    message
        .headers()
        .iter()
        .find(|header| header.name().as_str() == name)
        .and_then(|header| header.value().as_string().ok())
        .map(aws_smithy_types::str_bytes::StrBytes::as_str)
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use aws_smithy_types::event_stream::{Header, HeaderValue};

    /// One frame, written by `aws-smithy-eventstream`'s encoder.
    pub fn encode(headers: &[(&str, &str)], payload: &[u8]) -> Vec<u8> {
        let message = headers.iter().fold(
            Message::new(bytes::Bytes::copy_from_slice(payload)),
            |message, (name, value)| {
                message.add_header(Header::new(
                    (*name).to_owned(),
                    HeaderValue::String((*value).to_owned().into()),
                ))
            },
        );
        let mut frame = Vec::new();
        if let Err(error) = aws_smithy_eventstream::frame::write_message_to(&message, &mut frame) {
            panic!("encoding a test frame: {error}");
        }
        frame
    }

    fn chunk_event(payload: &[u8]) -> Vec<u8> {
        encode(
            &[
                (":event-type", "chunk"),
                (":content-type", "application/json"),
                (":message-type", "event"),
            ],
            payload,
        )
    }

    #[test]
    fn decodes_consecutive_messages_from_one_buffer() -> anyhow::Result<()> {
        let mut decoder = Decoder::default();
        let mut bytes = chunk_event(br#"{"bytes":"e30="}"#);
        bytes.extend(chunk_event(br#"{"bytes":"W10="}"#));
        decoder.push(&bytes);

        let first = decoder
            .next_message()?
            .ok_or_else(|| anyhow::anyhow!("first"))?;
        assert_eq!(header(&first, ":event-type"), Some("chunk"));
        assert_eq!(header(&first, ":message-type"), Some("event"));
        assert_eq!(first.payload().as_ref(), br#"{"bytes":"e30="}"#);
        let second = decoder
            .next_message()?
            .ok_or_else(|| anyhow::anyhow!("second"))?;
        assert_eq!(second.payload().as_ref(), br#"{"bytes":"W10="}"#);
        assert!(decoder.next_message()?.is_none());
        assert_eq!(decoder.pending(), 0);
        Ok(())
    }

    #[test]
    fn a_message_split_at_every_byte_decodes_once_whole() -> anyhow::Result<()> {
        let bytes = chunk_event(br#"{"bytes":"eyJ0eXBlIjoicGluZyJ9"}"#);
        let mut decoder = Decoder::default();
        for (index, byte) in bytes.iter().enumerate() {
            decoder.push(std::slice::from_ref(byte));
            let message = decoder.next_message()?;
            if index + 1 < bytes.len() {
                assert!(message.is_none(), "decoded early at byte {index}");
            } else {
                let message = message.ok_or_else(|| anyhow::anyhow!("never decoded"))?;
                assert_eq!(
                    message.payload().as_ref(),
                    br#"{"bytes":"eyJ0eXBlIjoicGluZyJ9"}"#
                );
            }
        }
        Ok(())
    }

    #[test]
    fn a_corrupted_payload_fails_the_message_crc() {
        let mut bytes = chunk_event(br#"{"bytes":"e30="}"#);
        let payload_byte = bytes.len() - 6;
        bytes[payload_byte] ^= 0x01;
        let mut decoder = Decoder::default();
        decoder.push(&bytes);
        assert!(matches!(decoder.next_message(), Err(DecodeError::Frame(_))));
    }

    #[test]
    fn a_corrupted_prelude_fails_its_crc() {
        let mut bytes = chunk_event(br#"{"bytes":"e30="}"#);
        bytes[8] ^= 0x01;
        let mut decoder = Decoder::default();
        decoder.push(&bytes);
        assert!(matches!(decoder.next_message(), Err(DecodeError::Frame(_))));
    }

    #[test]
    fn an_impossible_length_fails_before_the_body_is_awaited() {
        let mut bytes = chunk_event(br#"{"bytes":"e30="}"#);
        bytes[0] = 0x7f;
        let mut decoder = Decoder::default();
        decoder.push(&bytes[..4]);
        assert!(matches!(
            decoder.next_message(),
            Err(DecodeError::Length(_))
        ));
    }

    #[test]
    fn an_exception_message_carries_its_type() -> anyhow::Result<()> {
        let bytes = encode(
            &[
                (":exception-type", "throttlingException"),
                (":content-type", "application/json"),
                (":message-type", "exception"),
            ],
            br#"{"message":"Too many requests, please wait before trying again."}"#,
        );
        let mut decoder = Decoder::default();
        decoder.push(&bytes);
        let message = decoder
            .next_message()?
            .ok_or_else(|| anyhow::anyhow!("no message"))?;
        assert_eq!(header(&message, ":message-type"), Some("exception"));
        assert_eq!(
            header(&message, ":exception-type"),
            Some("throttlingException")
        );
        Ok(())
    }

    /// One frame from `aws-smithy-eventstream` 0.60.21's encoder carrying
    /// string headers plus a bool, a byte, an int16, an int32, an int64, a
    /// byte array, a timestamp and a UUID.
    const SMITHY_ENCODED_MESSAGE: &[u8] = &[
        0x00, 0x00, 0x00, 0xda, 0x00, 0x00, 0x00, 0xaa, 0x90, 0x64, 0xf2, 0x67, 0x0b, 0x3a, 0x65,
        0x76, 0x65, 0x6e, 0x74, 0x2d, 0x74, 0x79, 0x70, 0x65, 0x07, 0x00, 0x05, 0x63, 0x68, 0x75,
        0x6e, 0x6b, 0x04, 0x66, 0x6c, 0x61, 0x67, 0x00, 0x03, 0x6f, 0x66, 0x66, 0x01, 0x04, 0x62,
        0x79, 0x74, 0x65, 0x02, 0xfd, 0x05, 0x73, 0x68, 0x6f, 0x72, 0x74, 0x03, 0xfe, 0xd4, 0x05,
        0x63, 0x6f, 0x75, 0x6e, 0x74, 0x04, 0x00, 0x01, 0x11, 0x70, 0x04, 0x6c, 0x6f, 0x6e, 0x67,
        0x05, 0xff, 0xff, 0xff, 0xfe, 0xd5, 0xfa, 0x0e, 0x00, 0x04, 0x62, 0x6c, 0x6f, 0x62, 0x06,
        0x00, 0x03, 0x00, 0x01, 0x02, 0x02, 0x61, 0x74, 0x08, 0x00, 0x00, 0x01, 0xa1, 0xe9, 0x60,
        0xd6, 0x00, 0x02, 0x69, 0x64, 0x09, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0x01,
        0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0x0d, 0x3a, 0x63, 0x6f, 0x6e, 0x74, 0x65, 0x6e,
        0x74, 0x2d, 0x74, 0x79, 0x70, 0x65, 0x07, 0x00, 0x10, 0x61, 0x70, 0x70, 0x6c, 0x69, 0x63,
        0x61, 0x74, 0x69, 0x6f, 0x6e, 0x2f, 0x6a, 0x73, 0x6f, 0x6e, 0x0d, 0x3a, 0x6d, 0x65, 0x73,
        0x73, 0x61, 0x67, 0x65, 0x2d, 0x74, 0x79, 0x70, 0x65, 0x07, 0x00, 0x05, 0x65, 0x76, 0x65,
        0x6e, 0x74, 0x7b, 0x22, 0x62, 0x79, 0x74, 0x65, 0x73, 0x22, 0x3a, 0x22, 0x65, 0x79, 0x4a,
        0x30, 0x65, 0x58, 0x42, 0x6c, 0x49, 0x6a, 0x6f, 0x69, 0x63, 0x47, 0x6c, 0x75, 0x5a, 0x79,
        0x4a, 0x39, 0x22, 0x7d, 0x05, 0xf3, 0x46, 0x8d,
    ];

    #[test]
    fn reads_a_frame_with_every_header_type() -> anyhow::Result<()> {
        let mut decoder = Decoder::default();
        decoder.push(SMITHY_ENCODED_MESSAGE);
        let message = decoder
            .next_message()?
            .ok_or_else(|| anyhow::anyhow!("no message"))?;
        assert_eq!(header(&message, ":message-type"), Some("event"));
        assert_eq!(header(&message, ":event-type"), Some("chunk"));
        assert_eq!(header(&message, "flag"), None);
        assert_eq!(
            message.payload().as_ref(),
            br#"{"bytes":"eyJ0eXBlIjoicGluZyJ9"}"#
        );
        Ok(())
    }
}
