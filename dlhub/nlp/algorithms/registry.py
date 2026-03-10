from dlhub.nlp.types import Builder

from . import (
    albert,
    attention_pool,
    bert,
    bigbird,
    bigru,
    bilstm,
    birnn,
    cnn_bilstm,
    dcnn,
    distilbert,
    dpcnn,
    fasttext,
    fnet,
    gcnn,
    gmlp,
    gpt,
    gru,
    inceptioncnn,
    indrnn,
    kmaxcnn,
    linformer,
    longformer,
    lstm,
    mctextcnn,
    mlpmixer,
    nystromformer,
    performer,
    pooling,
    qrnn,
    rcnn,
    rescnn,
    resmlp,
    rnn,
    self_attn_rnn,
    sru,
    synthesizer,
    t5,
    talkingheads,
    tcn,
    textcnn,
    textcnn2d,
    transformer,
    vdcnn,
    wavenet,
)


def _update_unique(dst: dict[str, Builder], src: dict[str, Builder], *, source: str) -> None:
    for k, v in src.items():
        if k in dst:
            raise RuntimeError(f"Duplicate NLP arch name: {k!r} (from {source})")
        dst[k] = v


def build_registry() -> dict[str, Builder]:
    r: dict[str, Builder] = {}

    _update_unique(r, pooling.registry(), source="pooling")
    _update_unique(r, attention_pool.registry(), source="attention_pool")
    _update_unique(r, fasttext.registry(), source="fasttext")

    _update_unique(r, textcnn.registry(), source="textcnn")
    _update_unique(r, textcnn2d.registry(), source="textcnn2d")
    _update_unique(r, mctextcnn.registry(), source="mctextcnn")
    _update_unique(r, dpcnn.registry(), source="dpcnn")
    _update_unique(r, dcnn.registry(), source="dcnn")
    _update_unique(r, vdcnn.registry(), source="vdcnn")
    _update_unique(r, tcn.registry(), source="tcn")
    _update_unique(r, gcnn.registry(), source="gcnn")
    _update_unique(r, rescnn.registry(), source="rescnn")
    _update_unique(r, inceptioncnn.registry(), source="inceptioncnn")
    _update_unique(r, wavenet.registry(), source="wavenet")
    _update_unique(r, kmaxcnn.registry(), source="kmaxcnn")
    _update_unique(r, rcnn.registry(), source="rcnn")
    _update_unique(r, cnn_bilstm.registry(), source="cnn_bilstm")

    _update_unique(r, rnn.registry(), source="rnn")
    _update_unique(r, gru.registry(), source="gru")
    _update_unique(r, lstm.registry(), source="lstm")
    _update_unique(r, birnn.registry(), source="birnn")
    _update_unique(r, bigru.registry(), source="bigru")
    _update_unique(r, bilstm.registry(), source="bilstm")

    _update_unique(r, sru.registry(), source="sru")
    _update_unique(r, indrnn.registry(), source="indrnn")
    _update_unique(r, qrnn.registry(), source="qrnn")
    _update_unique(r, self_attn_rnn.registry(), source="self_attn_rnn")

    _update_unique(r, transformer.registry(), source="transformer")
    _update_unique(r, bert.registry(), source="bert")
    _update_unique(r, albert.registry(), source="albert")
    _update_unique(r, distilbert.registry(), source="distilbert")
    _update_unique(r, gpt.registry(), source="gpt")
    _update_unique(r, t5.registry(), source="t5")

    _update_unique(r, fnet.registry(), source="fnet")
    _update_unique(r, gmlp.registry(), source="gmlp")
    _update_unique(r, mlpmixer.registry(), source="mlpmixer")
    _update_unique(r, resmlp.registry(), source="resmlp")

    _update_unique(r, linformer.registry(), source="linformer")
    _update_unique(r, performer.registry(), source="performer")
    _update_unique(r, longformer.registry(), source="longformer")
    _update_unique(r, bigbird.registry(), source="bigbird")
    _update_unique(r, nystromformer.registry(), source="nystromformer")
    _update_unique(r, synthesizer.registry(), source="synthesizer")
    _update_unique(r, talkingheads.registry(), source="talkingheads")

    return r


REGISTRY: dict[str, Builder] = build_registry()

__all__ = ["REGISTRY", "build_registry"]
