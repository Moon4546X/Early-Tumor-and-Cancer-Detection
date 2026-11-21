"""
One-time conversion: Keras 2.x .joblib brain models -> Keras 3.x .keras files.

Run once:  .venv\\Scripts\\python.exe convert_brain_models.py
"""

import io, json, os, sys, tempfile, types, zipfile
from pathlib import Path

import joblib
import numpy as np

ROOT   = Path(__file__).resolve().parent
MODELS = ROOT / "Models"

# ---------------------------------------------------------------------------
# 1. Capture raw .keras zip bytes from a joblib file via a pickle shim
# ---------------------------------------------------------------------------

_cap = [None]

def _install_shim():
    shim = types.ModuleType("keras.src.saving.pickle_utils")
    def _capture(b): _cap[0] = b; return None
    shim.deserialize_model_from_bytecode = _capture
    shim.serialize_model_to_bytecode = lambda m: b""
    sys.modules["keras.src.saving.pickle_utils"] = shim

def extract_bytecode(path):
    _cap[0] = None
    _install_shim()
    try: joblib.load(path)
    except Exception: pass
    if _cap[0] is None:
        raise RuntimeError(f"Could not extract bytecode from {path}")
    return _cap[0]


# ---------------------------------------------------------------------------
# 2. Parse config to understand outer model structure
# ---------------------------------------------------------------------------

def outer_layers(cfg):
    return cfg.get("config", {}).get("layers", [])

def print_arch(cfg):
    for L in outer_layers(cfg):
        print(f"    [{L.get('class_name','?')}]  name={L.get('name','?')!r}")


# ---------------------------------------------------------------------------
# 3a. Build MobileNetV2-based model (bt-cnn / bt-mobilenet)
# ---------------------------------------------------------------------------

def build_mobilenet_model(cfg, weights_bytes):
    import keras

    layers = outer_layers(cfg)

    # Collect Dense head specs
    dense_specs  = [{"units": L["config"].get("units"), "activation": L["config"].get("activation","linear"),
                     "use_bias": L["config"].get("use_bias", True), "name": L.get("name","dense")}
                    for L in layers if L.get("class_name") == "Dense"]
    dropout_rate = next((L["config"].get("rate", 0.2) for L in layers if L.get("class_name") == "Dropout"), 0.2)
    names        = {L.get("class_name"): L.get("name") for L in layers}

    if not dense_specs:
        dense_specs = [{"units": 1280, "activation": "relu",   "use_bias": True, "name": "dense_5"},
                       {"units":    4, "activation": "linear", "use_bias": True, "name": "dense_4"}]

    print(f"  dropout={dropout_rate}  head={[(d['units'], d['activation']) for d in dense_specs]}")

    inputs = keras.Input(shape=(180, 180, 3), name="input_2")

    # Preprocessing: Rescaling replaces the two TFOpLambda nodes (/ 127.5 - 1.0)
    x = keras.layers.Rescaling(scale=1./127.5, offset=-1., name="rescaling")(inputs)

    base = keras.applications.MobileNetV2(input_shape=(180, 180, 3), include_top=False,
                                          weights=None, name="mobilenetv2_1.00_224")
    base.trainable = False
    x = base(x, training=False)

    x = keras.layers.GlobalAveragePooling2D(name=names.get("GlobalAveragePooling2D","global_average_pooling2d"))(x)
    x = keras.layers.Dropout(dropout_rate, name=names.get("Dropout","dropout_1"))(x)
    x = keras.layers.Flatten(name=names.get("Flatten","flatten_2"))(x)
    for spec in dense_specs:
        x = keras.layers.Dense(**spec)(x)

    model = keras.Model(inputs, x)
    _load_weights_byname(model, weights_bytes)
    return model


# ---------------------------------------------------------------------------
# 3b. Build simple CNN model (bt-cnn2: Rescaling + Conv2D + Dense)
# ---------------------------------------------------------------------------

def build_simple_cnn(cfg, weights_bytes):
    """
    bt-cnn2 uses a plain sequential-style CNN. We read layer types+configs
    from the outer layers list and reconstruct using the Functional API.
    """
    import keras

    layers = outer_layers(cfg)
    if not layers:
        raise RuntimeError("No layers found in config")

    # First layer must be InputLayer
    inp_cfg = layers[0].get("config", {})
    shape   = inp_cfg.get("batch_input_shape", inp_cfg.get("batch_shape", [None, 180, 180, 3]))[1:]
    if not shape:
        shape = (180, 180, 3)
    inp_name = layers[0].get("name") or "input_1"
    x = inputs = keras.Input(shape=tuple(shape), name=inp_name)

    for L in layers[1:]:
        cls  = L.get("class_name", "")
        name = L.get("name") or None
        bcfg = L.get("config", {})

        try:
            layer = _make_layer(cls, name, bcfg)
            x = layer(x)
        except Exception as e:
            print(f"  Warning: skipping layer {cls!r} ({name!r}): {e}")

    model = keras.Model(inputs, x)
    _load_weights_byname(model, weights_bytes)
    return model


def _make_layer(cls, name, bcfg):
    import keras

    # Sanitise config
    safe = dict(bcfg)
    safe.pop("build_config", None)
    safe.pop("module", None)
    safe.pop("registered_name", None)

    if cls == "BatchNormalization":
        ax = safe.get("axis")
        if isinstance(ax, list) and len(ax) == 1:
            safe["axis"] = ax[0]
    if cls in ("DepthwiseConv2D", "SeparableConv2D"):
        safe.pop("groups", None)
    if name:
        safe["name"] = name

    layer_cls = getattr(keras.layers, cls, None)
    if layer_cls is None:
        raise ValueError(f"Unknown layer class {cls!r}")

    import inspect
    sig = inspect.signature(layer_cls.__init__)
    valid = set(sig.parameters) - {"self", "kwargs", "args"}
    if valid:
        safe = {k: v for k, v in safe.items() if k in valid}

    return layer_cls(**safe)


def _load_weights_byname(model, weights_bytes):
    wpath = tempfile.mktemp(suffix=".h5")
    try:
        with open(wpath, "wb") as fh: fh.write(weights_bytes)
        model.load_weights(wpath, by_name=True, skip_mismatch=True)
        print("  Weights loaded (by_name, skip_mismatch=True).")
    finally:
        try: os.unlink(wpath)
        except OSError: pass


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

BRAIN_MODELS = [
    ("bt-cnn.joblib",       "bt-cnn.keras",       "mobilenet"),
    ("bt-cnn2.joblib",      "bt-cnn2.keras",       "simple"),
    ("bt-mobilenet.joblib", "bt-mobilenet.keras",  "mobilenet"),
]


def main():
    import keras
    print(f"Keras {keras.__version__}\n")

    for src_name, dst_name, arch in BRAIN_MODELS:
        src_path = MODELS / src_name
        dst_path = MODELS / dst_name

        if dst_path.exists():
            print(f"[SKIP] {dst_name} already exists.")
            continue

        print(f"[CONVERTING] {src_name}  ({arch})")
        try:
            bytecode = extract_bytecode(src_path)
            print(f"  Extracted {len(bytecode):,} bytes")

            zf  = zipfile.ZipFile(io.BytesIO(bytecode))
            cfg = json.loads(zf.read("config.json"))
            w   = zf.read("model.weights.h5")
            print(f"  Weights file: {len(w):,} bytes")
            print_arch(cfg)

            if arch == "mobilenet":
                model = build_mobilenet_model(cfg, w)
            else:
                model = build_simple_cnn(cfg, w)

            # Sanity check
            dummy = np.zeros((1, 180, 180, 3), dtype="float32")
            out   = model.predict(dummy, verbose=0)
            print(f"  Sanity predict OK: shape={out.shape}")

            model.save(str(dst_path))
            print(f"  Saved -> {dst_path}\n")

        except Exception as exc:
            import traceback; traceback.print_exc()
            print(f"  FAILED: {exc}\n")


if __name__ == "__main__":
    main()
