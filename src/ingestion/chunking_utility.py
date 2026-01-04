import re
import spacy
from copy import deepcopy

nlp = spacy.load("en_core_web_sm")

DISCOURSE_STARTS = {
    "then",
    "while",
    "after",
    "however",
    "meanwhile",
    "therefore",
    "thus",
    "instead",
    "otherwise",
}

PRONOUN_POS = {"PRON"}
PUNCTUATION_REGEX = re.compile(r"^[\W_]+")


def is_bad_start(text: str) -> bool:
    """
    Returns True if the chunk is likely context-dependent
    and should be merged or dropped.
    """
    if not text:
        return True

    # Leading punctuation
    if PUNCTUATION_REGEX.match(text):
        return True

    doc = nlp(text)

    if not doc:
        return True

    first_token = doc[0]

    # Pronoun-start
    if first_token.pos_ in PRONOUN_POS:
        return True

    # Discourse markers
    if first_token.text.lower() in DISCOURSE_STARTS:
        return True

    # Verb-start without subject
    if first_token.pos_ in {"VERB", "AUX"}:
        return True

    return False


def has_named_entity(text: str) -> bool:
    """
    Checks whether chunk introduces at least one named entity.
    """
    doc = nlp(text)
    return any(
        ent.label_ in {"PERSON", "ORG", "GPE", "WORK_OF_ART"} for ent in doc.ents
    )


def normalize_text(text: str) -> str:
    """
    Strips leading punctuation and excessive whitespace.
    """
    text = text.strip()
    text = PUNCTUATION_REGEX.sub("", text)
    return text.strip()


def repair_and_filter_chunks(
    chunks,
    *,
    max_chars=1200,
    min_chars=40,
):
    """
    Repairs chunks produced by RecursiveCharacterTextSplitter.

    - merges context-dependent chunks into previous chunk
    - drops irreparable fragments
    - enforces semantic indexability

    Returns: List[Document]
    """

    repaired = []
    buffer_chunk = None

    for chunk in chunks:
        new_chunk = deepcopy(chunk)
        text = normalize_text(new_chunk.page_content)

        if len(text) < min_chars:
            continue

        bad_start = is_bad_start(text)
        entity_present = has_named_entity(text)

        if bad_start or not entity_present:
            if buffer_chunk:
                merged = buffer_chunk.page_content + " " + text
                if len(merged) <= max_chars:
                    buffer_chunk.page_content = merged
                    continue
            # Drop if cannot merge safely
            continue

        # Finalize previous buffer
        if buffer_chunk:
            repaired.append(buffer_chunk)

        # Start new buffer
        add_anchor_title = chunk.metadata["alt_title"]
        if add_anchor_title:
            text = f"Movie: {add_anchor_title}.\nPlot:\n{text}"

        new_chunk.page_content = text
        buffer_chunk = new_chunk

    if buffer_chunk:
        repaired.append(buffer_chunk)

    return repaired
