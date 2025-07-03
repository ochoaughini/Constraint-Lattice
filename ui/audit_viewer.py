import json
from difflib import ndiff

import streamlit as st


def load_audit_log(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def diff_text(a, b):
    return "\n".join(ndiff(a.split(), b.split()))


st.title("Constraint Lattice AuditTrace Viewer")
log_file = st.file_uploader("Upload an AuditTrace .jsonl file", type="jsonl")
if log_file:
    log_path = "/tmp/_audittrace.jsonl"
    with open(log_path, "wb") as f:
        f.write(log_file.read())
    audit = load_audit_log(log_path)
    st.write(f"Loaded {len(audit)} audit steps.")
    filters = st.multiselect(
        "Filter by constraint", sorted(set(e["constraint"] for e in audit))
    )
    filtered = [e for e in audit if not filters or e["constraint"] in filters]
    for i, entry in enumerate(filtered):
        title = (
            f"{i + 1}. {entry['constraint']}::{entry['method']} "
            f"({entry['elapsed_ms']:.2f} ms)"
        )
        with st.expander(title):
            st.markdown(f"**Timestamp:** {entry['timestamp']}")
            st.markdown("**Pre Text:**")
            st.code(entry["pre_text"])
            st.markdown("**Post Text:**")
            st.code(entry["post_text"])
            st.markdown("**Diff:**")
            st.code(diff_text(entry["pre_text"], entry["post_text"]))
    if st.button("Export Filtered Log"):
        st.download_button(
            "Download",
            "\n".join(json.dumps(e) for e in filtered),
            file_name="filtered_audit.jsonl",
        )
