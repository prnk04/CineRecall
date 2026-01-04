from langchain_core.documents import Document


def dict_to_document(d: dict) -> Document:
    return Document(
        page_content=d["page_content"],
        metadata=d.get("metadata", {}),
    )


def json_to_documents(data: list[dict]) -> list[Document]:
    return [dict_to_document(d) for d in data]
