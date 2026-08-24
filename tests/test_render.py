"""Slack rendering: Markdown in, mrkdwn out, nothing mangled in between."""

from __future__ import annotations

from leo.slack.render import blocks_for, chunks, clean_prompt, to_mrkdwn


def test_bold_becomes_single_asterisks() -> None:
    assert to_mrkdwn("**bold** text") == "*bold* text"
    assert to_mrkdwn("__bold__ text") == "*bold* text"


def test_italics_become_underscores() -> None:
    assert to_mrkdwn("*italic* text") == "_italic_ text"


def test_a_heading_becomes_bold_not_italic() -> None:
    """Headings convert to bold; the italic pass must not then re-convert them."""

    assert to_mrkdwn("## Section") == "*Section*"
    assert to_mrkdwn("# Big\n\nbody") == "*Big*\n\nbody"


def test_links_become_slack_link_syntax() -> None:
    assert to_mrkdwn("[Acme](https://acme.com)") == "<https://acme.com|Acme>"


def test_bullets_are_normalised() -> None:
    assert to_mrkdwn("- one\n* two\n+ three") == "• one\n• two\n• three"


def test_strikethrough_loses_a_tilde() -> None:
    assert to_mrkdwn("~~gone~~") == "~gone~"


def test_code_is_never_reinterpreted() -> None:
    source = "```\n**not bold** and *not italic*\n```"
    assert to_mrkdwn(source) == source
    assert to_mrkdwn("use `a*b` here") == "use `a*b` here"


def test_arithmetic_is_not_mistaken_for_italics() -> None:
    assert to_mrkdwn("2*3 and 4*5") == "2*3 and 4*5"


def test_empty_input_is_empty_output() -> None:
    assert to_mrkdwn("") == ""


def test_short_text_is_one_chunk() -> None:
    assert chunks("hello") == ["hello"]


def test_long_text_splits_on_paragraphs_under_the_limit() -> None:
    text = "\n\n".join("p" * 900 for _ in range(6))
    pieces = chunks(text, limit=2900)
    assert len(pieces) > 1
    assert all(len(piece) <= 2900 for piece in pieces)


def test_a_code_fence_is_never_split_across_chunks() -> None:
    fence = "```\n" + "\n".join(f"line {i}" for i in range(200)) + "\n```"
    pieces = chunks(f"intro\n\n{fence}\n\nafter", limit=1200)
    fenced = [piece for piece in pieces if piece.startswith("```")]
    assert fenced and fenced[0].endswith("```")


def test_a_single_oversize_paragraph_is_hard_split() -> None:
    pieces = chunks("x" * 7000, limit=2000)
    assert len(pieces) == 4
    assert all(len(piece) <= 2000 for piece in pieces)


def test_blocks_carry_mrkdwn_sections() -> None:
    blocks = blocks_for("hello")
    assert blocks == [{"type": "section", "text": {"type": "mrkdwn", "text": "hello"}}]


def test_mentions_are_stripped_from_the_prompt() -> None:
    assert clean_prompt("<@U123> what is up") == "what is up"


def test_labelled_links_keep_both_label_and_url() -> None:
    assert clean_prompt("read <https://a.com|Acme>") == "read Acme (https://a.com)"


def test_bare_links_survive_unwrapped() -> None:
    """ "Summarise this link" is a real request; the URL has to reach the model."""

    assert clean_prompt("summarise <https://a.com/page>") == "summarise https://a.com/page"


def test_channel_references_are_dropped() -> None:
    assert clean_prompt("<#C1|general> hi there") == "hi there"
