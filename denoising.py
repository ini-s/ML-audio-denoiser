import sys
from collections import Counter

import onnx
from onnx import shape_inference, numpy_helper

def fmt_shape(tensor_type):
    """Format tensor shape as [d0, d1, ...] handling symbolic dims."""
    if not tensor_type.HasField("shape"):
        return "[]"
    dims = []
    for d in tensor_type.shape.dim:
        if d.HasField("dim_value"):
            dims.append(str(d.dim_value))
        elif d.HasField("dim_param"):
            dims.append(d.dim_param)   # e.g., 'batch'
        else:
            dims.append("?")
    return "[" + ", ".join(dims) + "]"

def main(path):
    print(f"\n== Loading model: {path} ==")
    model = onnx.load(path)

    # Try to infer missing shapes for readability
    try:
        model = shape_inference.infer_shapes(model)
        print("Shape inference: OK")
    except Exception as e:
        print(f"Shape inference: FAILED ({e})")

    print("\n== Model IR / Opset ==")
    print(f" IR version: {model.ir_version}")
    for imp in model.opset_import:
        print(f" Opset domain='{imp.domain or 'ai.onnx'}', version={imp.version}")

    if model.producer_name or model.producer_version:
        print(f" Producer: {model.producer_name} {model.producer_version}")
    if model.doc_string:
        print(f" Doc string: {model.doc_string}")

    if model.metadata_props:
        print("\n== Metadata ==")
        for p in model.metadata_props:
            print(f" {p.key}: {p.value}")

    g = model.graph

    print("\n== Inputs ==")
    for i in g.input:
        t = i.type.tensor_type
        print(f" - {i.name}: dtype={t.elem_type}, shape={fmt_shape(t)}")

    print("\n== Outputs ==")
    for o in g.output:
        t = o.type.tensor_type
        print(f" - {o.name}: dtype={t.elem_type}, shape={fmt_shape(t)}")

    # Initializers (weights)
    print("\n== Initializers (weights) summary ==")
    weights = {init.name: numpy_helper.to_array(init) for init in g.initializer}
    print(f" Total initializers: {len(weights)}")
    for k in list(weights.keys())[:5]:
        arr = weights[k]
        print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")

    # Operator histogram
    op_hist = Counter([n.op_type for n in g.node])
    print("\n== Operators histogram ==")
    for op, cnt in op_hist.most_common():
        print(f" {op}: {cnt}")

    # Show first N nodes (with attributes)
    N = min(15, len(g.node))
    print(f"\n== First {N} nodes ==")
    for n in g.node[:N]:
        ins = ", ".join(n.input)
        outs = ", ".join(n.output)
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
        print(f" {n.name or '<noname>'} | {n.op_type}")
        print(f"   inputs : {ins}")
        print(f"   outputs: {outs}")
        if attrs:
            # Avoid printing raw numpy dtypes for brevity
            printable = {k: (str(v.dtype) if hasattr(v, "dtype") else v) for k, v in attrs.items()}
            print(f"   attrs  : {printable}")

    # Rough parameter count
    total_params = sum(arr.size for arr in weights.values())
    print(f"\n== Rough parameter count: {total_params:,} ==")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_onnx.py path/to/model.onnx")
        sys.exit(1)
    main(sys.argv[1])