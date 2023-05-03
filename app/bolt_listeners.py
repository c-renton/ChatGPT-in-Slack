import logging
import re
import time

from openai.error import Timeout
from slack_bolt import App, Ack, BoltContext, BoltResponse
from slack_bolt.request.payload_utils import is_event
from slack_sdk.web import WebClient

from app.env import (
    OPENAI_TIMEOUT_SECONDS,
    SYSTEM_TEXT,
    OPENAI_MODERATION,
    TRANSLATE_MARKDOWN,
    ENABLE_BUTTONS,
)
from app.i18n import translate
from app.openai_ops import (
    start_receiving_openai_response,
    format_openai_message_content,
    consume_openai_stream_to_write_reply,
    build_system_text,
    messages_within_context_window,
    check_moderation,
)
from app.slack_ops import (
    find_parent_message,
    is_no_mention_thread,
    post_wip_message,
    update_wip_message,
)

from app.utils import redact_string

#
# Listener functions
#


def just_ack(ack: Ack):
    ack()


TIMEOUT_ERROR_MESSAGE = (
    f":warning: Sorry! It looks like OpenAI didn't respond within {OPENAI_TIMEOUT_SECONDS} seconds. "
    "Please try again later. :bow:"
)
DEFAULT_LOADING_TEXT = ":hourglass_flowing_sand: Wait a second, please ..."

ACTION_KEEP = {
    "name": "keep",
    "text": "Keep",
    "type": "button",
    "value": "keep",
    "style": "primary",
}

ACTION_REGENERATE = {
    "name": "regenerate",
    "text": "Regenerate",
    "type": "button",
    "value": "regenerate",
    "style": "default",
}

ACTION_DELETE = {
    "name": "delete",
    "text": "Delete",
    "type": "button",
    "value": "delete",
    "style": "danger",
}


def respond_to_app_mention(
    context: BoltContext,
    payload: dict,
    client: WebClient,
    logger: logging.Logger,
):
    if payload.get("thread_ts") is not None:
        parent_message = find_parent_message(
            client, context.channel_id, payload.get("thread_ts")
        )
        if parent_message is not None:
            if is_no_mention_thread(context, parent_message):
                # The message event handler will reply to this
                return

    wip_reply = None
    # Replace placeholder for Slack user ID in the system prompt
    system_text = build_system_text(SYSTEM_TEXT, TRANSLATE_MARKDOWN, context)
    messages = [{"role": "system", "content": system_text}]

    openai_api_key = context.get("OPENAI_API_KEY")
    try:
        if openai_api_key is None:
            client.chat_postMessage(
                channel=context.channel_id,
                text="To use this app, please configure your OpenAI API key first",
            )
            return

        user_id = context.actor_user_id or context.user_id

        msg_text = ""
        if payload.get("thread_ts") is not None:
            # Mentioning the bot user in a thread
            replies_in_thread = client.conversations_replies(
                channel=context.channel_id,
                ts=payload.get("thread_ts"),
                include_all_metadata=True,
                limit=1000,
            ).get("messages", [])
            for reply in replies_in_thread:
                msg_text = f"<@{reply['user']}>: " + format_openai_message_content(
                    redact_string(reply.get("text")), TRANSLATE_MARKDOWN
                )
                messages.append(
                    {
                        "role": (
                            "assistant"
                            if reply["user"] == context.bot_user_id
                            else "user"
                        ),
                        "content": msg_text,
                    }
                )
        else:
            # Strip bot Slack user ID from initial message
            msg_text = re.sub(f"<@{context.bot_user_id}>\\s*", "", payload["text"])
            msg_text = redact_string(msg_text)
            messages.append(
                {
                    "role": "user",
                    "content": f"<@{user_id}>: "
                    + format_openai_message_content(msg_text, TRANSLATE_MARKDOWN),
                }
            )

        loading_text = translate(
            openai_api_key=openai_api_key, context=context, text=DEFAULT_LOADING_TEXT
        )

        if ENABLE_BUTTONS:
            attachments = [
                {
                    "color": "#3AA3E3",
                    "fallback": "Buttons to click on",
                    "callback_id": "attachment_callback",
                    "actions": [ACTION_KEEP, ACTION_REGENERATE, ACTION_DELETE],
                }
            ]
        else:
            attachments = []

        wip_reply = post_wip_message(
            client=client,
            channel=context.channel_id,
            thread_ts=payload["ts"],
            loading_text=loading_text,
            messages=messages,
            attachments=attachments,
            user=context.user_id,
        )

        (
            messages,
            num_context_tokens,
            max_context_tokens,
        ) = messages_within_context_window(messages, model=context["OPENAI_MODEL"])
        num_messages = len([msg for msg in messages if msg.get("role") != "system"])
        if num_messages == 0:
            update_wip_message(
                client=client,
                channel=context.channel_id,
                ts=wip_reply["message"]["ts"],
                text=f":warning: The previous message is too long ({num_context_tokens}/{max_context_tokens} prompt tokens).",
                messages=messages,
                attachments=attachments,
                user=context.user_id,
            )
        else:
            flagged = False
            if OPENAI_MODERATION:
                mod_response = check_moderation(
                    openai_api_key=openai_api_key, text=msg_text
                )
                logger.debug(mod_response)
                if mod_response["results"][0]["flagged"]:
                    flagged = True

            if flagged:
                update_wip_message(
                    client=client,
                    channel=context.channel_id,
                    ts=wip_reply["message"]["ts"],
                    text=":warning: The above message was flagged as violating the OpenAI usage policy",
                    messages=messages,
                    attachments=attachments,
                    user=context.user_id,
                )
            else:
                stream = start_receiving_openai_response(
                    openai_api_key=openai_api_key,
                    model=context["OPENAI_MODEL"],
                    temperature=context["OPENAI_TEMPERATURE"],
                    messages=messages,
                    user=context.user_id,
                    openai_api_type=context["OPENAI_API_TYPE"],
                    openai_api_base=context["OPENAI_API_BASE"],
                    openai_api_version=context["OPENAI_API_VERSION"],
                    openai_deployment_id=context["OPENAI_DEPLOYMENT_ID"],
                )
                consume_openai_stream_to_write_reply(
                    client=client,
                    wip_reply=wip_reply,
                    context=context,
                    user_id=user_id,
                    messages=messages,
                    attachments=attachments,
                    stream=stream,
                    timeout_seconds=OPENAI_TIMEOUT_SECONDS,
                    openai_moderation=OPENAI_MODERATION,
                    enable_buttons=ENABLE_BUTTONS,
                    translate_markdown=TRANSLATE_MARKDOWN,
                    logger=logger,
                )

    except Timeout:
        if wip_reply is not None:
            text = (
                (
                    wip_reply.get("message", {}).get("text", "")
                    if wip_reply is not None
                    else ""
                )
                + "\n\n"
                + translate(
                    openai_api_key=openai_api_key,
                    context=context,
                    text=TIMEOUT_ERROR_MESSAGE,
                )
            )
            client.chat_update(
                channel=context.channel_id,
                ts=wip_reply["message"]["ts"],
                text=text,
            )
    except Exception as e:
        text = (
            (
                wip_reply.get("message", {}).get("text", "")
                if wip_reply is not None
                else ""
            )
            + "\n\n"
            + translate(
                openai_api_key=openai_api_key,
                context=context,
                text=f":warning: Failed to start a conversation with ChatGPT: {e}",
            )
        )
        logger.exception(text, e)
        if wip_reply is not None:
            client.chat_update(
                channel=context.channel_id,
                ts=wip_reply["message"]["ts"],
                text=text,
            )


def respond_to_new_message(
    context: BoltContext,
    payload: dict,
    client: WebClient,
    logger: logging.Logger,
):
    if payload.get("bot_id") is not None and payload.get("bot_id") != context.bot_id:
        # Skip a new message by a different app
        return

    wip_reply = None
    try:
        is_in_dm_with_bot = payload.get("channel_type") == "im"
        is_no_mention_required = False
        thread_ts = payload.get("thread_ts")
        if is_in_dm_with_bot is False and thread_ts is None:
            return

        openai_api_key = context.get("OPENAI_API_KEY")
        if openai_api_key is None:
            return

        messages_in_context = []
        if is_in_dm_with_bot is True and thread_ts is None:
            # In the DM with the bot
            past_messages = client.conversations_history(
                channel=context.channel_id,
                include_all_metadata=True,
                limit=100,
            ).get("messages", [])
            past_messages.reverse()
            # Remove old messages
            for message in past_messages:
                seconds = time.time() - float(message.get("ts"))
                if seconds < 86400:  # less than 1 day
                    messages_in_context.append(message)
            is_no_mention_required = True
        else:
            # In a thread with the bot in a channel
            messages_in_context = client.conversations_replies(
                channel=context.channel_id,
                ts=thread_ts,
                include_all_metadata=True,
                limit=1000,
            ).get("messages", [])
            if is_in_dm_with_bot is True:
                is_no_mention_required = True
            else:
                the_parent_message_found = False
                for message in messages_in_context:
                    if message.get("ts") == thread_ts:
                        the_parent_message_found = True
                        is_no_mention_required = is_no_mention_thread(context, message)
                        break
                if the_parent_message_found is False:
                    parent_message = find_parent_message(
                        client, context.channel_id, thread_ts
                    )
                    if parent_message is not None:
                        is_no_mention_required = is_no_mention_thread(
                            context, parent_message
                        )

        messages = []
        user_id = context.actor_user_id or context.user_id
        last_assistant_idx = -1
        indices_to_remove = []
        for idx, reply in enumerate(messages_in_context):
            maybe_event_type = reply.get("metadata", {}).get("event_type")
            if maybe_event_type == "chat-gpt-convo":
                if context.bot_id != reply.get("bot_id"):
                    # Remove messages by a different app
                    indices_to_remove.append(idx)
                    continue
                maybe_new_messages = (
                    reply.get("metadata", {}).get("event_payload", {}).get("messages")
                )
                if maybe_new_messages is not None:
                    if len(messages) == 0 or user_id is None:
                        new_user_id = (
                            reply.get("metadata", {})
                            .get("event_payload", {})
                            .get("user")
                        )
                        if new_user_id is not None:
                            user_id = new_user_id
                    messages = maybe_new_messages
                    last_assistant_idx = idx

        if is_no_mention_required is False:
            return

        if is_in_dm_with_bot is True or last_assistant_idx == -1:
            # To know whether this app needs to start a new convo
            if not next(filter(lambda msg: msg["role"] == "system", messages), None):
                # Replace placeholder for Slack user ID in the system prompt
                system_text = build_system_text(
                    SYSTEM_TEXT, TRANSLATE_MARKDOWN, context
                )
                messages.insert(0, {"role": "system", "content": system_text})

        filtered_messages_in_context = []
        for idx, reply in enumerate(messages_in_context):
            # Strip bot Slack user ID from initial message
            if idx == 0:
                reply["text"] = re.sub(
                    f"<@{context.bot_user_id}>\\s*", "", reply["text"]
                )
            if idx not in indices_to_remove:
                filtered_messages_in_context.append(reply)
        if len(filtered_messages_in_context) == 0:
            return

        msg_text = ""
        for reply in filtered_messages_in_context:
            msg_user_id = reply.get("user")
            msg_text = f"<@{msg_user_id}>: " + format_openai_message_content(
                redact_string(reply.get("text")), TRANSLATE_MARKDOWN
            )
            messages.append(
                {
                    "content": msg_text,
                    "role": "user",
                }
            )

        loading_text = translate(
            openai_api_key=openai_api_key, context=context, text=DEFAULT_LOADING_TEXT
        )

        if ENABLE_BUTTONS:
            attachments = [
                {
                    "color": "#3AA3E3",
                    "fallback": "Buttons to click on",
                    "callback_id": "attachment_callback",
                    "actions": [ACTION_KEEP, ACTION_REGENERATE, ACTION_DELETE],
                }
            ]
        else:
            attachments = []

        wip_reply = post_wip_message(
            client=client,
            channel=context.channel_id,
            thread_ts=payload.get("thread_ts") if is_in_dm_with_bot else payload["ts"],
            loading_text=loading_text,
            messages=messages,
            attachments=attachments,
            user=user_id,
        )

        (
            messages,
            num_context_tokens,
            max_context_tokens,
        ) = messages_within_context_window(messages, model=context["OPENAI_MODEL"])
        num_messages = len([msg for msg in messages if msg.get("role") != "system"])
        if num_messages == 0:
            update_wip_message(
                client=client,
                channel=context.channel_id,
                ts=wip_reply["message"]["ts"],
                text=f":warning: The previous message is too long ({num_context_tokens}/{max_context_tokens} prompt tokens).",
                messages=messages,
                attachments=attachments,
                user=context.user_id,
            )
        else:
            flagged = False
            if OPENAI_MODERATION:
                mod_response = check_moderation(
                    openai_api_key=openai_api_key, text=msg_text
                )
                logger.debug(mod_response)
                if mod_response["results"][0]["flagged"]:
                    flagged = True

            if flagged:
                update_wip_message(
                    client=client,
                    channel=context.channel_id,
                    ts=wip_reply["message"]["ts"],
                    text=":warning: The above message was flagged as violating the OpenAI usage policy",
                    messages=messages,
                    attachments=attachments,
                    user=user_id,
                )
            else:
                stream = start_receiving_openai_response(
                    openai_api_key=openai_api_key,
                    model=context["OPENAI_MODEL"],
                    temperature=context["OPENAI_TEMPERATURE"],
                    messages=messages,
                    user=user_id,
                    openai_api_type=context["OPENAI_API_TYPE"],
                    openai_api_base=context["OPENAI_API_BASE"],
                    openai_api_version=context["OPENAI_API_VERSION"],
                    openai_deployment_id=context["OPENAI_DEPLOYMENT_ID"],
                )

                latest_replies = client.conversations_replies(
                    channel=context.channel_id,
                    ts=wip_reply.get("ts"),
                    include_all_metadata=True,
                    limit=1000,
                )
                if (
                    latest_replies.get("messages", [])[-1]["ts"]
                    != wip_reply["message"]["ts"]
                ):
                    # Since a new reply will come soon, this app abandons this reply
                    client.chat_delete(
                        channel=context.channel_id,
                        ts=wip_reply["message"]["ts"],
                    )
                    return

                consume_openai_stream_to_write_reply(
                    client=client,
                    wip_reply=wip_reply,
                    context=context,
                    user_id=user_id,
                    messages=messages,
                    attachments=attachments,
                    stream=stream,
                    timeout_seconds=OPENAI_TIMEOUT_SECONDS,
                    openai_moderation=OPENAI_MODERATION,
                    enable_buttons=ENABLE_BUTTONS,
                    translate_markdown=TRANSLATE_MARKDOWN,
                    logger=logger,
                )

    except Timeout:
        if wip_reply is not None:
            text = (
                (
                    wip_reply.get("message", {}).get("text", "")
                    if wip_reply is not None
                    else ""
                )
                + "\n\n"
                + translate(
                    openai_api_key=openai_api_key,
                    context=context,
                    text=TIMEOUT_ERROR_MESSAGE,
                )
            )
            client.chat_update(
                channel=context.channel_id,
                ts=wip_reply["message"]["ts"],
                text=text,
            )
    except Exception as e:
        text = (
            (
                wip_reply.get("message", {}).get("text", "")
                if wip_reply is not None
                else ""
            )
            + "\n\n"
            + f":warning: Failed to reply: {e}"
        )
        logger.exception(text, e)
        if wip_reply is not None:
            client.chat_update(
                channel=context.channel_id,
                ts=wip_reply["message"]["ts"],
                text=text,
            )


def regenerate_reply(
    context: BoltContext,
    payload: dict,
    is_in_dm_with_bot: bool,
    client: WebClient,
    logger: logging.Logger,
):
    wip_reply = None
    try:
        is_no_mention_required = False
        thread_ts = payload.get("thread_ts")
        if is_in_dm_with_bot is False and thread_ts is None:
            return

        openai_api_key = context.get("OPENAI_API_KEY")
        if openai_api_key is None:
            return

        messages_in_context = []
        if is_in_dm_with_bot is True and thread_ts is None:
            # In the DM with the bot
            past_messages = client.conversations_history(
                channel=context.channel_id,
                include_all_metadata=True,
                limit=100,
            ).get("messages", [])
            past_messages.reverse()
            # Remove old messages
            for message in past_messages:
                seconds = time.time() - float(message.get("ts"))
                if seconds < 86400:  # less than 1 day
                    messages_in_context.append(message)
            is_no_mention_required = True
        else:
            # In a thread with the bot in a channel
            messages_in_context = client.conversations_replies(
                channel=context.channel_id,
                ts=thread_ts,
                include_all_metadata=True,
                limit=1000,
            ).get("messages", [])
            if is_in_dm_with_bot is True:
                is_no_mention_required = True
            else:
                the_parent_message_found = False
                for message in messages_in_context:
                    if message.get("ts") == thread_ts:
                        the_parent_message_found = True
                        is_no_mention_required = is_no_mention_thread(context, message)
                        break
                if the_parent_message_found is False:
                    parent_message = find_parent_message(
                        client, context.channel_id, thread_ts
                    )
                    if parent_message is not None:
                        is_no_mention_required = is_no_mention_thread(
                            context, parent_message
                        )

        messages = []
        user_id = context.actor_user_id or context.user_id
        last_assistant_idx = -1
        indices_to_remove = []
        for idx, reply in enumerate(messages_in_context):
            if reply.get("ts") >= payload.get("ts"):
                # Remove messages not before this one
                indices_to_remove.append(idx)
                continue
            maybe_event_type = reply.get("metadata", {}).get("event_type")
            if maybe_event_type == "chat-gpt-convo":
                if context.bot_id != reply.get("bot_id"):
                    # Remove messages by a different app
                    indices_to_remove.append(idx)
                    continue

        if is_in_dm_with_bot is True:
            # To know whether this app needs to start a new convo
            if not next(filter(lambda msg: msg["role"] == "system", messages), None):
                # Replace placeholder for Slack user ID in the system prompt
                system_text = build_system_text(
                    SYSTEM_TEXT, TRANSLATE_MARKDOWN, context
                )
                messages.insert(0, {"role": "system", "content": system_text})

        filtered_messages_in_context = []
        for idx, reply in enumerate(messages_in_context):
            # Strip bot Slack user ID from initial message
            if idx == 0:
                reply["text"] = reply["text"].replace(f"<@{context.bot_user_id}>", "")
            if idx not in indices_to_remove:
                filtered_messages_in_context.append(reply)
        if len(filtered_messages_in_context) == 0:
            return

        msg_text = ""
        for reply in filtered_messages_in_context:
            msg_user_id = reply.get("user")
            msg_text = f"<@{msg_user_id}>: " + format_openai_message_content(
                redact_string(reply.get("text")), TRANSLATE_MARKDOWN
            )
            messages.append(
                {
                    "content": msg_text,
                    "role": "user",
                }
            )

        loading_text = translate(
            openai_api_key=openai_api_key, context=context, text=DEFAULT_LOADING_TEXT
        )

        wip_reply = update_wip_message(
            client=client,
            channel=context.channel_id,
            ts=payload["ts"],
            text=loading_text,
            messages=messages,
            attachments=payload["attachments"],
            user=user_id,
        )
        wip_reply["message"]["ts"] = payload["ts"]

        flagged = False
        if OPENAI_MODERATION:
            mod_response = check_moderation(
                openai_api_key=openai_api_key, text=msg_text
            )
            logger.debug(mod_response)
            if mod_response["results"][0]["flagged"]:
                flagged = True

        if flagged:
            update_wip_message(
                client=client,
                channel=context.channel_id,
                ts=wip_reply["message"]["ts"],
                text=":warning: The above message was flagged as violating the OpenAI usage policy",
                messages=messages,
                attachments=payload["attachments"],
                user=user_id,
            )
        else:
            stream = start_receiving_openai_response(
                openai_api_key=openai_api_key,
                model=context["OPENAI_MODEL"],
                temperature=context["OPENAI_TEMPERATURE"],
                messages=messages,
                user=user_id,
                openai_api_type=context["OPENAI_API_TYPE"],
                openai_api_base=context["OPENAI_API_BASE"],
                openai_api_version=context["OPENAI_API_VERSION"],
                openai_deployment_id=context["OPENAI_DEPLOYMENT_ID"],
            )

            consume_openai_stream_to_write_reply(
                client=client,
                wip_reply=wip_reply,
                context=context,
                user_id=user_id,
                messages=messages,
                attachments=payload["attachments"],
                stream=stream,
                timeout_seconds=OPENAI_TIMEOUT_SECONDS,
                openai_moderation=OPENAI_MODERATION,
                enable_buttons=ENABLE_BUTTONS,
                translate_markdown=TRANSLATE_MARKDOWN,
                logger=logger,
            )

    except Timeout:
        if wip_reply is not None:
            text = (
                (
                    wip_reply.get("message", {}).get("text", "")
                    if wip_reply is not None
                    else ""
                )
                + "\n\n"
                + translate(
                    openai_api_key=openai_api_key,
                    context=context,
                    text=TIMEOUT_ERROR_MESSAGE,
                )
            )
            client.chat_update(
                channel=context.channel_id,
                ts=wip_reply["message"]["ts"],
                text=text,
            )
    except Exception as e:
        text = (
            (
                wip_reply.get("message", {}).get("text", "")
                if wip_reply is not None
                else ""
            )
            + "\n\n"
            + f":warning: Failed to reply: {e}"
        )
        logger.exception(text, e)
        if wip_reply is not None:
            client.chat_update(
                channel=context.channel_id,
                ts=wip_reply["message"]["ts"],
                text=text,
            )


def handle_attachments(
    context: BoltContext,
    body: dict,
    payload: dict,
    client: WebClient,
    logger: logging.Logger,
):
    # Get the channel and message timestamp from the payload
    channel = body["channel"]["id"]
    message_ts = body["message_ts"]

    if payload["type"] == "button":
        if payload["value"] == "keep":
            # Delete the attachments data
            client.chat_update(
                channel=channel,
                ts=message_ts,
                text=body["original_message"]["text"],
                attachments=[],
            )
        elif payload["value"] == "regenerate":
            is_in_dm_with_bot = body["channel"]["name"] == "directmessage"
            this_msg = []
            if is_in_dm_with_bot:
                this_msg = body["original_message"]
            else:
                # Fetch the the current message in the thread
                thread_ts = body["original_message"]["thread_ts"]
                response = client.conversations_replies(
                    channel=channel,
                    ts=thread_ts,
                    include_all_metadata=True,
                    limit=1000,
                )
                messages = response["messages"]
                for msg in messages:
                    if msg["ts"] == message_ts:
                        this_msg = msg
                        break
                if this_msg is None:
                    return
            regenerate_reply(
                context=context,
                payload=this_msg,
                is_in_dm_with_bot=is_in_dm_with_bot,
                client=client,
                logger=logger,
            )
        elif payload["value"] == "delete":
            # Delete the current message
            client.chat_delete(
                channel=channel,
                ts=message_ts,
            )


def register_listeners(app: App):
    app.event("app_mention")(ack=just_ack, lazy=[respond_to_app_mention])
    app.event("message")(ack=just_ack, lazy=[respond_to_new_message])
    app.action("attachment_callback")(ack=just_ack, lazy=[handle_attachments])


MESSAGE_SUBTYPES_TO_SKIP = ["message_changed", "message_deleted"]


# To reduce unnecessary workload in this app,
# this before_authorize function skips message changed/deleted events.
# Especially, "message_changed" events can be triggered many times when the app rapidly updates its reply.
def before_authorize(
    body: dict,
    payload: dict,
    logger: logging.Logger,
    next_,
):
    if (
        is_event(body)
        and payload.get("type") == "message"
        and payload.get("subtype") in MESSAGE_SUBTYPES_TO_SKIP
    ):
        logger.debug(
            "Skipped the following middleware and listeners "
            f"for this message event (subtype: {payload.get('subtype')})"
        )
        return BoltResponse(status=200, body="")
    next_()
