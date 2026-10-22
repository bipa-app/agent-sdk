//! What a model can be shown as an image or a PDF: content checks shared by
//! the tools that produce media and the providers that send it.

use base64::Engine as _;

/// The media type `prefix` starts with: PNG, JPEG, GIF, WebP or PDF.
#[must_use]
pub fn detect_media_magic(prefix: &[u8]) -> Option<&'static str> {
    if prefix.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some("image/png")
    } else if prefix.starts_with(b"\xff\xd8\xff") {
        Some("image/jpeg")
    } else if prefix.starts_with(b"GIF87a") || prefix.starts_with(b"GIF89a") {
        Some("image/gif")
    } else if prefix.len() >= 12 && &prefix[..4] == b"RIFF" && &prefix[8..12] == b"WEBP" {
        Some("image/webp")
    } else if prefix.starts_with(b"%PDF-") {
        Some("application/pdf")
    } else {
        None
    }
}

/// Longest side, in pixels, of an image providers accept (Anthropic's limit).
pub const MAX_IMAGE_SIDE_PX: u32 = 8000;

/// An image's width and height from its header, without decoding it, or
/// `None` when the header cannot be read.
#[must_use]
pub fn image_dimensions(bytes: &[u8], media_type: &str) -> Option<(u32, u32)> {
    let be16 = |at: usize| {
        Some(u32::from(u16::from_be_bytes(
            bytes.get(at..at + 2)?.try_into().ok()?,
        )))
    };
    let le16 = |at: usize| {
        Some(u32::from(u16::from_le_bytes(
            bytes.get(at..at + 2)?.try_into().ok()?,
        )))
    };
    let le24 = |at: usize| {
        let b = bytes.get(at..at + 3)?;
        Some(u32::from(b[0]) | (u32::from(b[1]) << 8) | (u32::from(b[2]) << 16))
    };
    let be32 = |at: usize| Some(u32::from_be_bytes(bytes.get(at..at + 4)?.try_into().ok()?));
    match media_type {
        "image/png" => (bytes.get(12..16)? == b"IHDR").then(|| Some((be32(16)?, be32(20)?)))?,
        "image/gif" => Some((le16(6)?, le16(8)?)),
        "image/webp" => match bytes.get(12..16)? {
            b"VP8 " => Some((le16(26)? & 0x3fff, le16(28)? & 0x3fff)),
            b"VP8L" => {
                let b = bytes.get(21..25)?;
                let width = 1 + (u32::from(b[0]) | (u32::from(b[1] & 0x3f) << 8));
                let height = 1
                    + ((u32::from(b[1]) >> 6)
                        | (u32::from(b[2]) << 2)
                        | (u32::from(b[3] & 0x0f) << 10));
                Some((width, height))
            }
            b"VP8X" => Some((le24(24)? + 1, le24(27)? + 1)),
            _ => None,
        },
        "image/jpeg" => {
            let mut at = 2;
            loop {
                while *bytes.get(at)? == 0xff && *bytes.get(at + 1)? == 0xff {
                    at += 1;
                }
                if *bytes.get(at)? != 0xff {
                    return None;
                }
                let marker = *bytes.get(at + 1)?;
                let is_frame =
                    matches!(marker, 0xc0..=0xcf) && !matches!(marker, 0xc4 | 0xc8 | 0xcc);
                if is_frame {
                    return Some((be16(at + 7)?, be16(at + 5)?));
                }
                if matches!(marker, 0x01 | 0xd0..=0xd9) {
                    at += 2;
                    continue;
                }
                at += 2 + usize::try_from(be16(at + 2)?).ok()?;
            }
        }
        _ => None,
    }
}

/// Why base64 `data` cannot be shown to a model as the image `media_type`.
///
/// `None` when it can: it decodes, its content is that image type, and its
/// header gives dimensions within [`MAX_IMAGE_SIDE_PX`].
///
/// Only as much is decoded as the header needs (a JPEG's frame header can
/// sit past its metadata), so a large image costs a few kilobytes. A verdict
/// that needed more is remembered: a history sends the same image again on
/// every request.
#[must_use]
pub fn image_problem(media_type: &str, data: &str) -> Option<String> {
    // Base64 is ASCII, so slicing bytes is exact; non-ASCII fails to decode.
    let data = data.as_bytes();
    if let Some(Verdict(problem)) = check_prefix(media_type, data, FIRST_PREFIX) {
        return problem;
    }
    let cache = &*VERDICTS;
    let key = cache.key(media_type, data);
    if let Some(Verdict(problem)) = cache.lookup(&key, data) {
        return problem;
    }
    let mut prefix = FIRST_PREFIX;
    loop {
        prefix *= 16;
        if let Some(verdict) = check_prefix(media_type, data, prefix) {
            cache.store(key, data, prefix.min(data.len()), verdict.clone());
            return verdict.0;
        }
    }
}

/// [`image_problem`]'s answer once the header has been read.
#[derive(Clone)]
struct Verdict(Option<String>);

/// The base64 characters decoded first; most headers sit well inside.
const FIRST_PREFIX: usize = 4 * 1024;

/// The verdict on `data`'s first `prefix` characters, or `None` while its
/// header lies further on. Always a verdict once `prefix` covers all of it.
fn check_prefix(media_type: &str, data: &[u8], prefix: usize) -> Option<Verdict> {
    let whole = prefix >= data.len();
    let chunk = if whole {
        data
    } else {
        &data[..prefix - prefix % 4]
    };
    let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(chunk) else {
        return Some(Verdict(Some("its data is not valid base64".to_owned())));
    };
    match detect_media_magic(&bytes) {
        Some(found) if found == media_type => {}
        Some(found) => {
            return Some(Verdict(Some(format!(
                "its content is {found}, not {media_type}"
            ))));
        }
        None if whole || bytes.len() >= 16 => {
            return Some(Verdict(Some(format!(
                "its content is not a {media_type} image"
            ))));
        }
        None => {}
    }
    if let Some((width, height)) = image_dimensions(&bytes, media_type) {
        return Some(Verdict((width.max(height) > MAX_IMAGE_SIDE_PX).then(
            || format!("it is {width}x{height} px, over {MAX_IMAGE_SIDE_PX} px on a side"),
        )));
    }
    whole.then(|| {
        Verdict(Some(format!(
            "its {media_type} header has no readable dimensions"
        )))
    })
}

static VERDICTS: std::sync::LazyLock<VerdictCache> = std::sync::LazyLock::new(VerdictCache::new);

/// Verdicts that needed more than [`FIRST_PREFIX`], keyed by the media type,
/// the length and a hash of the first [`FIRST_PREFIX`] characters. A hit is
/// trusted only when the hash of the prefix its verdict was computed from
/// matches too: the verdict depends on nothing else. The hash is seeded per
/// process, so content can't be made to collide with an image seen earlier.
struct VerdictCache {
    seed: u64,
    entries: std::sync::Mutex<std::collections::HashMap<VerdictKey, CachedVerdict>>,
}

#[derive(PartialEq, Eq, Hash)]
struct VerdictKey {
    media_type: String,
    len: usize,
    head: u64,
}

struct CachedVerdict {
    decoded: usize,
    digest: u64,
    verdict: Verdict,
}

/// Entries kept before the cache starts over; each is a few dozen bytes.
const MAX_CACHED_VERDICTS: usize = 4096;

impl VerdictCache {
    fn new() -> Self {
        use std::hash::BuildHasher as _;
        Self {
            seed: std::collections::hash_map::RandomState::new().hash_one(0_u8),
            entries: std::sync::Mutex::default(),
        }
    }

    fn key(&self, media_type: &str, data: &[u8]) -> VerdictKey {
        VerdictKey {
            media_type: media_type.to_owned(),
            len: data.len(),
            head: self.digest(data, FIRST_PREFIX),
        }
    }

    fn digest(&self, data: &[u8], prefix: usize) -> u64 {
        xxhash_rust::xxh64::xxh64(&data[..prefix.min(data.len())], self.seed)
    }

    fn lookup(&self, key: &VerdictKey, data: &[u8]) -> Option<Verdict> {
        let (decoded, digest, verdict) = self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(key)
            .map(|entry| (entry.decoded, entry.digest, entry.verdict.clone()))?;
        (self.digest(data, decoded) == digest).then_some(verdict)
    }

    fn store(&self, key: VerdictKey, data: &[u8], decoded: usize, verdict: Verdict) {
        let digest = self.digest(data, decoded);
        let mut entries = self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if entries.len() >= MAX_CACHED_VERDICTS {
            entries.clear();
        }
        entries.insert(
            key,
            CachedVerdict {
                decoded,
                digest,
                verdict,
            },
        );
    }

    #[cfg(test)]
    fn holds(&self, media_type: &str, data: &str) -> bool {
        let key = self.key(media_type, data.as_bytes());
        self.entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains_key(&key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png(width: u32, height: u32, len: usize) -> Vec<u8> {
        let mut bytes = b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR".to_vec();
        bytes.extend_from_slice(&width.to_be_bytes());
        bytes.extend_from_slice(&height.to_be_bytes());
        bytes.resize(len.max(bytes.len()), 0);
        bytes
    }

    fn encoded(bytes: &[u8]) -> String {
        base64::engine::general_purpose::STANDARD.encode(bytes)
    }

    #[test]
    fn a_real_image_of_its_declared_type_passes() {
        assert_eq!(
            image_problem("image/png", &encoded(&png(640, 480, 64))),
            None
        );
        assert_eq!(
            image_problem("image/png", &encoded(&png(1920, 1080, 3_000_000))),
            None
        );
    }

    #[test]
    fn content_that_is_not_its_declared_image_is_named() {
        let svg = encoded(b"<svg xmlns='http://www.w3.org/2000/svg'/>");
        assert!(
            image_problem("image/png", &svg).is_some_and(|why| why.contains("not a image/png"))
        );
        let jpeg = encoded(&[
            0xff, 0xd8, 0xff, 0xe0, 0, 4, 0, 0, 0xff, 0xc0, 0, 8, 8, 1, 0xe0, 2, 0x80,
        ]);
        assert!(image_problem("image/png", &jpeg).is_some_and(|why| why.contains("image/jpeg")));
        assert!(image_problem("image/png", &"A".repeat(4096)).is_some());
        assert!(
            image_problem("image/png", "not base64!").is_some_and(|why| why.contains("base64"))
        );
    }

    #[test]
    fn an_image_too_large_or_without_dimensions_is_named() {
        assert!(
            image_problem("image/png", &encoded(&png(9000, 400, 64)))
                .is_some_and(|why| why.contains("9000x400"))
        );
        assert!(
            image_problem("image/png", &encoded(b"\x89PNG\r\n\x1a\n"))
                .is_some_and(|why| why.contains("no readable dimensions"))
        );
    }

    /// A JPEG whose frame header (`width` x `height`) follows 50 KB of
    /// metadata, past the first prefix the check decodes.
    fn jpeg_behind_metadata(width: u16, height: u16) -> Vec<u8> {
        let mut jpeg = vec![0xff, 0xd8, 0xff, 0xe1, 0xc3, 0x52];
        jpeg.resize(jpeg.len() + 0xc350, 0);
        jpeg.extend_from_slice(&[0xff, 0xc0, 0, 8, 8]);
        jpeg.extend_from_slice(&height.to_be_bytes());
        jpeg.extend_from_slice(&width.to_be_bytes());
        jpeg
    }

    #[test]
    fn a_jpeg_frame_header_past_its_metadata_is_found() {
        assert_eq!(
            image_problem("image/jpeg", &encoded(&jpeg_behind_metadata(640, 480))),
            None
        );
    }

    #[test]
    fn non_ascii_data_is_not_base64_wherever_it_falls() {
        let straddling_first_prefix = format!("{}{}", "A".repeat(4095), "\u{20ac}".repeat(10));
        let mut header = encoded(&jpeg_behind_metadata(640, 480));
        header.truncate(65_535);
        let straddling_second_prefix = format!("{header}{}", "\u{20ac}".repeat(10));
        for data in [straddling_first_prefix, straddling_second_prefix] {
            assert_eq!(
                image_problem("image/jpeg", &data).as_deref(),
                Some("its data is not valid base64"),
            );
        }
    }

    #[test]
    fn a_remembered_verdict_is_only_reused_for_the_same_image() {
        let small = encoded(&jpeg_behind_metadata(640, 480));
        let huge = encoded(&jpeg_behind_metadata(9000, 10));
        assert_eq!(small.len(), huge.len());
        assert_eq!(small[..FIRST_PREFIX], huge[..FIRST_PREFIX]);

        assert_eq!(image_problem("image/jpeg", &small), None);
        assert!(VERDICTS.holds("image/jpeg", &small), "the verdict is kept");
        assert!(
            image_problem("image/jpeg", &huge).is_some_and(|why| why.contains("9000x10")),
            "another image with the same head and length is checked anew",
        );
        assert_eq!(image_problem("image/jpeg", &small), None);
    }
}
