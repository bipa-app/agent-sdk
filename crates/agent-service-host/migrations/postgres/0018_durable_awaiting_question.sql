-- ENG-9279: first-class durable AwaitingQuestion and AnswerQuestion.

ALTER TABLE agent_sdk_tasks
    DROP CONSTRAINT agent_sdk_tasks_status_check,
    DROP CONSTRAINT agent_sdk_tasks_waiting_state_check;

ALTER TABLE agent_sdk_tasks
    ADD CONSTRAINT agent_sdk_tasks_status_check
        CHECK (
            status IN (
                'queued', 'pending', 'running', 'waiting_on_children',
                'awaiting_confirmation', 'awaiting_question',
                'completed', 'failed', 'cancelled'
            )
        ),
    ADD CONSTRAINT agent_sdk_tasks_waiting_state_check
        CHECK (
            jsonb_typeof(state_json) = 'object'
            AND state_json ? 'kind'
            AND state_json ->> 'kind' IN (
                'none', 'waiting_on_children', 'awaiting_confirmation',
                'awaiting_question', 'answered_question',
                'subagent_invocation', 'ready_to_resume'
            )
            AND (
                (
                    state_json ->> 'kind' IN ('waiting_on_children', 'subagent_invocation')
                    AND status = 'waiting_on_children'
                    AND pending_child_count > 0
                )
                OR (
                    state_json ->> 'kind' = 'subagent_invocation'
                    AND status IN ('pending', 'running')
                    AND pending_child_count = 0
                )
                OR (
                    state_json ->> 'kind' = 'awaiting_confirmation'
                    AND status = 'awaiting_confirmation'
                    AND pending_child_count = 0
                )
                OR (
                    state_json ->> 'kind' = 'awaiting_question'
                    AND status = 'awaiting_question'
                    AND pending_child_count = 0
                )
                OR (
                    state_json ->> 'kind' IN ('ready_to_resume', 'answered_question')
                    AND status IN ('pending', 'running')
                    AND pending_child_count = 0
                )
                OR (
                    state_json ->> 'kind' = 'none'
                    AND status IN (
                        'queued', 'pending', 'running',
                        'completed', 'failed', 'cancelled'
                    )
                    AND pending_child_count = 0
                )
            )
        );

DROP INDEX agent_sdk_tasks_root_admission_slot_idx;
CREATE UNIQUE INDEX agent_sdk_tasks_root_admission_slot_idx
    ON agent_sdk_tasks (thread_id)
    WHERE kind = 'root_turn'
      AND status IN (
          'pending', 'running', 'waiting_on_children',
          'awaiting_confirmation', 'awaiting_question'
      );

ALTER TABLE agent_sdk_idempotency
    DROP CONSTRAINT agent_sdk_idempotency_kind_check;
ALTER TABLE agent_sdk_idempotency
    ADD CONSTRAINT agent_sdk_idempotency_kind_check
        CHECK (kind IN (
            'create_thread', 'submit_work', 'fork_thread',
            'decide_confirmation', 'answer_question'
        ));
