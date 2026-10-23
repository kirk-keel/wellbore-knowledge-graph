"""Generates a clean synthetic engineering incident report PDF for testing."""

from pathlib import Path


def create_sample_pdf(output_path: str = "data/sample_docs/sample_bop_incident.pdf"):
    content = (
        "BT\n"
        "/F1 12 Tf\n"
        "50 720 Td\n"
        "(INCIDENT INVESTIGATION REPORT: TITAN SUBSEA BOP FAILURE) Tj\n"
        "0 -25 Td\n"
        "(Well: Deepwater Titan Well | Location: Gulf of Mexico) Tj\n"
        "0 -20 Td\n"
        "(Penetrating Norphlet Formation Sandstone at 16,100 ft) Tj\n"
        "0 -20 Td\n"
        "(Asset Serial Number: BOP-HYD-5000-08 | Rating: 15,000 psi) Tj\n"
        "0 -20 Td\n"
        "(Observed Failure Mode: FAIL-RAMLEAK ram leak during gas influx) Tj\n"
        "0 -20 Td\n"
        "(Total NPT: 48.5 hours incurred) Tj\n"
        "0 -20 Td\n"
        "(Mandatory Remediation: SOP-SHUTIN-01 and API-RP-53 compliance) Tj\n"
        "ET\n"
    )

    stream_bytes = content.encode("latin1")
    length = len(stream_bytes)

    header = b"%PDF-1.4\n"
    obj1 = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    obj2 = b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    obj3 = b"3 0 obj\n<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >>\nendobj\n"
    obj4 = b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    obj5 = f"5 0 obj\n<< /Length {length} >>\nstream\n".encode("latin1") + stream_bytes + b"\nendstream\nendobj\n"

    offsets = [0]
    pos = len(header)
    offsets.append(pos)
    pos += len(obj1)
    offsets.append(pos)
    pos += len(obj2)
    offsets.append(pos)
    pos += len(obj3)
    offsets.append(pos)
    pos += len(obj4)
    offsets.append(pos)
    pos += len(obj5)

    xref = f"xref\n0 6\n0000000000 65535 f \n".encode("latin1")
    for off in offsets[1:]:
        xref += f"{off:010d} 00000 n \n".encode("latin1")

    trailer = f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{pos}\n%%EOF\n".encode("latin1")

    full_pdf = header + obj1 + obj2 + obj3 + obj4 + obj5 + xref + trailer

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(full_pdf)


if __name__ == "__main__":
    create_sample_pdf()
    print("Sample PDF generated successfully.")
