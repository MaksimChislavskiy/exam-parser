from pathlib import Path

import fitz


def pdf_to_images(
    pdf_path: str,
    output_dir: str,
    dpi: int = 300
) -> list[str]:

    output_path = Path(output_dir)

    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    doc = fitz.open(pdf_path)

    result = []

    zoom = dpi / 72
    matrix = fitz.Matrix(
        zoom,
        zoom
    )

    for page_num in range(len(doc)):

        page = doc[page_num]

        pix = page.get_pixmap(
            matrix=matrix,
            alpha=False
        )

        filename = (
            output_path /
            f"page_{page_num + 1}.png"
        )

        pix.save(str(filename))

        result.append(
            str(filename)
        )

        print(
            f"Saved: {filename}"
        )

    doc.close()

    return result
