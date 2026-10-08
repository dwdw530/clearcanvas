"""Remove explicitly marked PDF watermarks without touching shared resources."""

from pypdf.generic import (
    ArrayObject,
    ContentStream,
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
)


def _watermark_tag(operands, resources):
    if not operands:
        return False
    if operands[0] == "/Watermark":
        return True
    if operands[0] != "/Artifact" or len(operands) < 2:
        return False
    properties = operands[1]
    if isinstance(properties, NameObject):
        properties = (
            resources.get("/Properties", DictionaryObject()).get_object().get(properties, {})
        )
    properties = properties.get_object() if hasattr(properties, "get_object") else properties
    return isinstance(properties, dict) and properties.get("/Subtype") == "/Watermark"


def _isolated(operations):
    # Only drop blocks whose graphic/text state cannot leak into later content.
    if not operations or operations[0][1] != b"q" or operations[-1][1] != b"Q":
        return False
    depth, text = 0, 0
    for index, (_, operator) in enumerate(operations):
        depth += (operator == b"q") - (operator == b"Q")
        text += (operator == b"BT") - (operator == b"ET")
        if depth < 0 or text not in (0, 1) or (depth == 0 and index < len(operations) - 1):
            return False
        if operator in (b"Tj", b"TJ", b"'", b'"') and not text:
            return False
        if operator in (
            b"m",
            b"l",
            b"c",
            b"v",
            b"y",
            b"h",
            b"re",
            b"W",
            b"W*",
            b"S",
            b"s",
            b"f",
            b"F",
            b"f*",
            b"B",
            b"B*",
            b"b",
            b"b*",
            b"n",
        ):
            return False
    return depth == 0 and text == 0


def _clean_stream(stream, resources, pdf, ancestors, depth, cancelled):
    from clearcanvas.documents import DocumentError

    if cancelled():
        raise DocumentError("已取消。", "cancelled")
    if depth > 8 or id(stream) in ancestors:
        return None, resources, 0
    content = ContentStream(stream, pdf)
    operations = content.operations
    if len(operations) > 100_000:
        return None, resources, 0
    ancestors = ancestors | {id(stream)}
    output, removed = [], 0
    index, in_text = 0, False
    while index < len(operations):
        if index % 256 == 0 and cancelled():
            raise DocumentError("已取消。", "cancelled")
        operands, operator = operations[index]
        if not in_text and operator in (b"BDC", b"BMC") and _watermark_tag(operands, resources):
            end, nesting = index + 1, 1
            while end < len(operations) and nesting:
                op = operations[end][1]
                nesting += (op in (b"BDC", b"BMC")) - (op == b"EMC")
                end += 1
            if nesting == 0 and _isolated(operations[index + 1 : end - 1]):
                removed += 1
                index = end
                continue
            # Some writers put q/Q around the marked-content wrapper instead
            # of inside it. Remove that entire state-isolated unit, including
            # the q already appended, only when no unmarked operation intervenes.
            if (
                nesting == 0 and output and output[-1][1] == b"q"
                and end < len(operations) and operations[end][1] == b"Q"
                and _isolated([output[-1], *operations[index + 1 : end - 1], operations[end]])
            ):
                output.pop()
                removed += 1
                index = end + 1
                continue
        output.append((operands, operator))
        if operator in (b"BT", b"ET"):
            in_text = operator == b"BT"
        index += 1

    xobjects = resources.get("/XObject")
    changed_resources = DictionaryObject(resources)
    if xobjects:
        xobjects = xobjects.get_object()
        replacements = DictionaryObject(xobjects)
        names = {args[0] for args, op in output if op == b"Do" and args}
        for name in names:
            if name not in xobjects:
                continue
            form = xobjects[name].get_object()
            if form.get("/Subtype") != "/Form":
                continue
            form_resources = form.get("/Resources", resources).get_object()
            cleaned, nested_resources, count = _clean_stream(
                form, form_resources, pdf, ancestors, depth + 1, cancelled
            )
            if count:
                clone = DecodedStreamObject()
                clone.update(
                    {
                        k: v
                        for k, v in form.items()
                        if k not in ("/Length", "/Filter", "/DecodeParms")
                    }
                )
                clone.set_data(cleaned.get_data())
                clone[NameObject("/Resources")] = nested_resources
                replacements[name] = pdf._add_object(clone)
                removed += count
        changed_resources[NameObject("/XObject")] = replacements
    if not removed:
        return None, resources, 0
    content.operations = output
    return content, changed_resources, removed


def remove_pdf_watermarks(page, cancelled=lambda: False) -> int:
    """The page must belong to a PdfWriter; unmarked content is never guessed."""
    pdf = page.indirect_reference.pdf
    resources = page.get("/Resources", DictionaryObject()).get_object()
    stream = page.get_contents()
    removed = 0
    if stream is not None:
        cleaned, resources, removed = _clean_stream(stream, resources, pdf, set(), 0, cancelled)
        if removed:
            # replace_contents may reuse the original indirect stream's object ID.
            # A fresh reference is required when other pages share that stream.
            cleaned.indirect_reference = None
            page[NameObject("/Contents")] = pdf._add_object(cleaned)
            page[NameObject("/Resources")] = resources
    annotations = page.get("/Annots")
    if annotations:
        kept = ArrayObject()
        for reference in annotations.get_object():
            if reference.get_object().get("/Subtype") == "/Watermark":
                removed += 1
            else:
                kept.append(reference)
        page[NameObject("/Annots")] = kept
    return removed
