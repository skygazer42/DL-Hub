from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    arch: str = (
        "dncnn"  # dncnn | edsr | rrdbnet | carn | resunet | unet3plus | r2unet | denseunet | brdnet | attention_unet | unetpp | mwcnn | hinet | ircnn | jorder | rescan | prenet | nlrn | scunet | convnext_unet | aspp_unet | cbam_unet | restormer | noise2noise_unet | bm3d | median_filter | wiener_filter | guided_filter | bilateral_filter | non_local_means | total_variation | anisotropic_diffusion | wavelet_shrinkage | anscombe_wiener | lee_filter | kuan_filter | stripe_remover | dead_hot_pixel_corrector | line_defect_corrector | rowcol_bias_corrector | block_bias_corrector | debanding_filter | ffdnet | nafnet | drunet | swinir | ridnet | ddpm_unet | mirnet | mprnet | uformer | cbdnet | didn | rcan | rdn | memnet | drrn | rednet | pridnet | dhdn | bsn | dbsn | pixelcnn_bsn | gated_pixelcnn_bsn
    )
    variant: str = "dncnn_9"
    in_channels: int = 1
    sigma: float = 0.1  # for BM3D baseline


def list_supported_arches() -> list[str]:
    from dlhub.vision.denoising.anisotropic_diffusion import _VARIANTS as anisodiff_variants
    from dlhub.vision.denoising.anscombe_wiener import _VARIANTS as anscombe_wiener_variants
    from dlhub.vision.denoising.aspp_unet import _VARIANTS as aspp_unet_variants
    from dlhub.vision.denoising.attention_unet import _VARIANTS as attention_unet_variants
    from dlhub.vision.denoising.bilateral_filter import _VARIANTS as bilateral_variants
    from dlhub.vision.denoising.block_bias_corrector import (
        _VARIANTS as block_bias_corrector_variants,
    )
    from dlhub.vision.denoising.bm3d import _VARIANTS as bm3d_variants
    from dlhub.vision.denoising.brdnet import _VARIANTS as brdnet_variants
    from dlhub.vision.denoising.bsn import _VARIANTS as bsn_variants
    from dlhub.vision.denoising.carn import _VARIANTS as carn_variants
    from dlhub.vision.denoising.cbam_unet import _VARIANTS as cbam_unet_variants
    from dlhub.vision.denoising.cbdnet import _VARIANTS as cbdnet_variants
    from dlhub.vision.denoising.convnext_unet import _VARIANTS as convnext_unet_variants
    from dlhub.vision.denoising.dbsn import _VARIANTS as dbsn_variants
    from dlhub.vision.denoising.ddn import _VARIANTS as ddn_variants
    from dlhub.vision.denoising.ddpm_unet import _VARIANTS as ddpm_unet_variants
    from dlhub.vision.denoising.dead_hot_pixel_corrector import (
        _VARIANTS as dead_hot_pixel_corrector_variants,
    )
    from dlhub.vision.denoising.derainformer import _VARIANTS as derainformer_variants
    from dlhub.vision.denoising.debanding_filter import _VARIANTS as debanding_filter_variants
    from dlhub.vision.denoising.denseunet import _VARIANTS as denseunet_variants
    from dlhub.vision.denoising.dhdn import _VARIANTS as dhdn_variants
    from dlhub.vision.denoising.did_mdn import _VARIANTS as did_mdn_variants
    from dlhub.vision.denoising.didn import _VARIANTS as didn_variants
    from dlhub.vision.denoising.dncnn import _VARIANTS as dncnn_variants
    from dlhub.vision.denoising.drrn import _VARIANTS as drrn_variants
    from dlhub.vision.denoising.drunet import _VARIANTS as drunet_variants
    from dlhub.vision.denoising.edsr import _VARIANTS as edsr_variants
    from dlhub.vision.denoising.ffdnet import _VARIANTS as ffdnet_variants
    from dlhub.vision.denoising.gated_pixelcnn_bsn import _VARIANTS as gated_pixelcnn_bsn_variants
    from dlhub.vision.denoising.guided_filter import _VARIANTS as guided_filter_variants
    from dlhub.vision.denoising.hinet import _VARIANTS as hinet_variants
    from dlhub.vision.denoising.ircnn import _VARIANTS as ircnn_variants
    from dlhub.vision.denoising.jorder import _VARIANTS as jorder_variants
    from dlhub.vision.denoising.kuan_filter import _VARIANTS as kuan_filter_variants
    from dlhub.vision.denoising.lee_filter import _VARIANTS as lee_filter_variants
    from dlhub.vision.denoising.line_defect_corrector import (
        _VARIANTS as line_defect_corrector_variants,
    )
    from dlhub.vision.denoising.median_filter import _VARIANTS as median_filter_variants
    from dlhub.vision.denoising.memnet import _VARIANTS as memnet_variants
    from dlhub.vision.denoising.mirnet import _VARIANTS as mirnet_variants
    from dlhub.vision.denoising.mprnet import _VARIANTS as mprnet_variants
    from dlhub.vision.denoising.mwcnn import _VARIANTS as mwcnn_variants
    from dlhub.vision.denoising.nafnet import _VARIANTS as nafnet_variants
    from dlhub.vision.denoising.nlrn import _VARIANTS as nlrn_variants
    from dlhub.vision.denoising.noise2noise import _VARIANTS as n2n_variants
    from dlhub.vision.denoising.non_local_means import _VARIANTS as non_local_means_variants
    from dlhub.vision.denoising.pixelcnn_bsn import _VARIANTS as pixelcnn_bsn_variants
    from dlhub.vision.denoising.prenet import _VARIANTS as prenet_variants
    from dlhub.vision.denoising.pridnet import _VARIANTS as pridnet_variants
    from dlhub.vision.denoising.r2unet import _VARIANTS as r2unet_variants
    from dlhub.vision.denoising.rcan import _VARIANTS as rcan_variants
    from dlhub.vision.denoising.rcdnet import _VARIANTS as rcdnet_variants
    from dlhub.vision.denoising.rdn import _VARIANTS as rdn_variants
    from dlhub.vision.denoising.rednet import _VARIANTS as rednet_variants
    from dlhub.vision.denoising.rescan import _VARIANTS as rescan_variants
    from dlhub.vision.denoising.restormer import _VARIANTS as restormer_variants
    from dlhub.vision.denoising.resunet import _VARIANTS as resunet_variants
    from dlhub.vision.denoising.ridnet import _VARIANTS as ridnet_variants
    from dlhub.vision.denoising.rowcol_bias_corrector import (
        _VARIANTS as rowcol_bias_corrector_variants,
    )
    from dlhub.vision.denoising.rrdbnet import _VARIANTS as rrdbnet_variants
    from dlhub.vision.denoising.scunet import _VARIANTS as scunet_variants
    from dlhub.vision.denoising.stripe_remover import _VARIANTS as stripe_remover_variants
    from dlhub.vision.denoising.swinir import _VARIANTS as swinir_variants
    from dlhub.vision.denoising.spanet import _VARIANTS as spanet_variants
    from dlhub.vision.denoising.total_variation import _VARIANTS as total_variation_variants
    from dlhub.vision.denoising.transweather import _VARIANTS as transweather_variants
    from dlhub.vision.denoising.uformer import _VARIANTS as uformer_variants
    from dlhub.vision.denoising.unet3plus import _VARIANTS as unet3plus_variants
    from dlhub.vision.denoising.unetpp import _VARIANTS as unetpp_variants
    from dlhub.vision.denoising.wavelet_shrinkage import _VARIANTS as wavelet_shrinkage_variants
    from dlhub.vision.denoising.wiener_filter import _VARIANTS as wiener_filter_variants

    out: list[str] = []
    out.extend([f"dncnn:{k}" for k in sorted(dncnn_variants)])
    out.extend([f"edsr:{k}" for k in sorted(edsr_variants)])
    out.extend([f"rrdbnet:{k}" for k in sorted(rrdbnet_variants)])
    out.extend([f"carn:{k}" for k in sorted(carn_variants)])
    out.extend([f"resunet:{k}" for k in sorted(resunet_variants)])
    out.extend([f"unet3plus:{k}" for k in sorted(unet3plus_variants)])
    out.extend([f"r2unet:{k}" for k in sorted(r2unet_variants)])
    out.extend([f"denseunet:{k}" for k in sorted(denseunet_variants)])
    out.extend([f"brdnet:{k}" for k in sorted(brdnet_variants)])
    out.extend([f"attention_unet:{k}" for k in sorted(attention_unet_variants)])
    out.extend([f"unetpp:{k}" for k in sorted(unetpp_variants)])
    out.extend([f"mwcnn:{k}" for k in sorted(mwcnn_variants)])
    out.extend([f"hinet:{k}" for k in sorted(hinet_variants)])
    out.extend([f"ircnn:{k}" for k in sorted(ircnn_variants)])
    out.extend([f"jorder:{k}" for k in sorted(jorder_variants)])
    out.extend([f"ddn:{k}" for k in sorted(ddn_variants)])
    out.extend([f"spanet:{k}" for k in sorted(spanet_variants)])
    out.extend([f"did_mdn:{k}" for k in sorted(did_mdn_variants)])
    out.extend([f"rcdnet:{k}" for k in sorted(rcdnet_variants)])
    out.extend([f"transweather:{k}" for k in sorted(transweather_variants)])
    out.extend([f"derainformer:{k}" for k in sorted(derainformer_variants)])
    out.extend([f"rescan:{k}" for k in sorted(rescan_variants)])
    out.extend([f"prenet:{k}" for k in sorted(prenet_variants)])
    out.extend([f"nlrn:{k}" for k in sorted(nlrn_variants)])
    out.extend([f"scunet:{k}" for k in sorted(scunet_variants)])
    out.extend([f"convnext_unet:{k}" for k in sorted(convnext_unet_variants)])
    out.extend([f"aspp_unet:{k}" for k in sorted(aspp_unet_variants)])
    out.extend([f"cbam_unet:{k}" for k in sorted(cbam_unet_variants)])
    out.extend([f"restormer:{k}" for k in sorted(restormer_variants)])
    out.extend([f"nafnet:{k}" for k in sorted(nafnet_variants)])
    out.extend([f"swinir:{k}" for k in sorted(swinir_variants)])
    out.extend([f"ridnet:{k}" for k in sorted(ridnet_variants)])
    out.extend([f"ffdnet:{k}" for k in sorted(ffdnet_variants)])
    out.extend([f"drunet:{k}" for k in sorted(drunet_variants)])
    out.extend([f"noise2noise_unet:{k}" for k in sorted(n2n_variants)])
    out.extend([f"bm3d:{k}" for k in sorted(bm3d_variants)])
    out.extend([f"median_filter:{k}" for k in sorted(median_filter_variants)])
    out.extend([f"wiener_filter:{k}" for k in sorted(wiener_filter_variants)])
    out.extend([f"guided_filter:{k}" for k in sorted(guided_filter_variants)])
    out.extend([f"bilateral_filter:{k}" for k in sorted(bilateral_variants)])
    out.extend([f"non_local_means:{k}" for k in sorted(non_local_means_variants)])
    out.extend([f"total_variation:{k}" for k in sorted(total_variation_variants)])
    out.extend([f"anisotropic_diffusion:{k}" for k in sorted(anisodiff_variants)])
    out.extend([f"wavelet_shrinkage:{k}" for k in sorted(wavelet_shrinkage_variants)])
    out.extend([f"anscombe_wiener:{k}" for k in sorted(anscombe_wiener_variants)])
    out.extend([f"lee_filter:{k}" for k in sorted(lee_filter_variants)])
    out.extend([f"kuan_filter:{k}" for k in sorted(kuan_filter_variants)])
    out.extend([f"stripe_remover:{k}" for k in sorted(stripe_remover_variants)])
    out.extend([f"dead_hot_pixel_corrector:{k}" for k in sorted(dead_hot_pixel_corrector_variants)])
    out.extend([f"line_defect_corrector:{k}" for k in sorted(line_defect_corrector_variants)])
    out.extend([f"rowcol_bias_corrector:{k}" for k in sorted(rowcol_bias_corrector_variants)])
    out.extend([f"block_bias_corrector:{k}" for k in sorted(block_bias_corrector_variants)])
    out.extend([f"debanding_filter:{k}" for k in sorted(debanding_filter_variants)])
    out.extend([f"bsn:{k}" for k in sorted(bsn_variants)])
    out.extend([f"cbdnet:{k}" for k in sorted(cbdnet_variants)])
    out.extend([f"ddpm_unet:{k}" for k in sorted(ddpm_unet_variants)])
    out.extend([f"dbsn:{k}" for k in sorted(dbsn_variants)])
    out.extend([f"didn:{k}" for k in sorted(didn_variants)])
    out.extend([f"mirnet:{k}" for k in sorted(mirnet_variants)])
    out.extend([f"mprnet:{k}" for k in sorted(mprnet_variants)])
    out.extend([f"rcan:{k}" for k in sorted(rcan_variants)])
    out.extend([f"rdn:{k}" for k in sorted(rdn_variants)])
    out.extend([f"memnet:{k}" for k in sorted(memnet_variants)])
    out.extend([f"drrn:{k}" for k in sorted(drrn_variants)])
    out.extend([f"rednet:{k}" for k in sorted(rednet_variants)])
    out.extend([f"pridnet:{k}" for k in sorted(pridnet_variants)])
    out.extend([f"dhdn:{k}" for k in sorted(dhdn_variants)])
    out.extend([f"uformer:{k}" for k in sorted(uformer_variants)])
    out.extend([f"pixelcnn_bsn:{k}" for k in sorted(pixelcnn_bsn_variants)])
    out.extend([f"gated_pixelcnn_bsn:{k}" for k in sorted(gated_pixelcnn_bsn_variants)])
    return out


def build_model(cfg: ModelConfig) -> nn.Module:
    arch_raw = str(cfg.arch).strip()
    arch = arch_raw.lower()

    # Allow either "dncnn" + variant, or single string like "dncnn:dncnn_17".
    variant = str(cfg.variant).strip()
    if ":" in arch_raw:
        pref, name = arch_raw.split(":", 1)
        arch = pref.strip().lower()
        variant = name.strip()

    in_channels = int(cfg.in_channels)

    if arch in {"dncnn"}:
        from dlhub.vision.denoising.dncnn import build_dncnn_denoiser

        return build_dncnn_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"edsr"}:
        from dlhub.vision.denoising.edsr import build_edsr_denoiser

        return build_edsr_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"rrdbnet"}:
        from dlhub.vision.denoising.rrdbnet import build_rrdbnet_denoiser

        return build_rrdbnet_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"carn"}:
        from dlhub.vision.denoising.carn import build_carn_denoiser

        return build_carn_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"resunet"}:
        from dlhub.vision.denoising.resunet import build_resunet_denoiser

        return build_resunet_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"unet3plus", "unet3+", "unet3_plus"}:
        from dlhub.vision.denoising.unet3plus import build_unet3plus_denoiser

        return build_unet3plus_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"r2unet", "r2u", "r2u_net"}:
        from dlhub.vision.denoising.r2unet import build_r2unet_denoiser

        return build_r2unet_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"denseunet", "dense_u_net"}:
        from dlhub.vision.denoising.denseunet import build_denseunet_denoiser

        return build_denseunet_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"brdnet"}:
        from dlhub.vision.denoising.brdnet import build_brdnet_denoiser

        return build_brdnet_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"attention_unet", "attn_unet", "attention"}:
        from dlhub.vision.denoising.attention_unet import build_attention_unet_denoiser

        return build_attention_unet_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"unetpp", "unet++", "unet_plus_plus"}:
        from dlhub.vision.denoising.unetpp import build_unetpp_denoiser

        return build_unetpp_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"mwcnn"}:
        from dlhub.vision.denoising.mwcnn import build_mwcnn_denoiser

        return build_mwcnn_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"hinet"}:
        from dlhub.vision.denoising.hinet import build_hinet_denoiser

        return build_hinet_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"ircnn"}:
        from dlhub.vision.denoising.ircnn import build_ircnn_denoiser

        return build_ircnn_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"jorder"}:
        from dlhub.vision.denoising.jorder import build_jorder_denoiser

        return build_jorder_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"ddn"}:
        from dlhub.vision.denoising.ddn import build_ddn_denoiser

        return build_ddn_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"spanet", "spa_net"}:
        from dlhub.vision.denoising.spanet import build_spanet_denoiser

        return build_spanet_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"did_mdn", "did-mdn", "didmdn"}:
        from dlhub.vision.denoising.did_mdn import build_did_mdn_denoiser

        return build_did_mdn_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"rcdnet"}:
        from dlhub.vision.denoising.rcdnet import build_rcdnet_denoiser

        return build_rcdnet_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"transweather", "trans_weather"}:
        from dlhub.vision.denoising.transweather import build_transweather_denoiser

        return build_transweather_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"derainformer", "derain_former"}:
        from dlhub.vision.denoising.derainformer import build_derainformer_denoiser

        return build_derainformer_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"rescan"}:
        from dlhub.vision.denoising.rescan import build_rescan_denoiser

        return build_rescan_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"prenet", "pre_net"}:
        from dlhub.vision.denoising.prenet import build_prenet_denoiser

        return build_prenet_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"nlrn"}:
        from dlhub.vision.denoising.nlrn import build_nlrn_denoiser

        return build_nlrn_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"scunet"}:
        from dlhub.vision.denoising.scunet import build_scunet_denoiser

        return build_scunet_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"convnext_unet", "convnextunet"}:
        from dlhub.vision.denoising.convnext_unet import build_convnext_unet_denoiser

        return build_convnext_unet_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"aspp_unet", "asppunet"}:
        from dlhub.vision.denoising.aspp_unet import build_aspp_unet_denoiser

        return build_aspp_unet_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"cbam_unet", "cbamunet"}:
        from dlhub.vision.denoising.cbam_unet import build_cbam_unet_denoiser

        return build_cbam_unet_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"restormer"}:
        from dlhub.vision.denoising.restormer import build_restormer_denoiser

        return build_restormer_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"nafnet"}:
        from dlhub.vision.denoising.nafnet import build_nafnet_denoiser

        return build_nafnet_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"swinir"}:
        from dlhub.vision.denoising.swinir import build_swinir_denoiser

        return build_swinir_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"ridnet"}:
        from dlhub.vision.denoising.ridnet import build_ridnet_denoiser

        return build_ridnet_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"ffdnet"}:
        from dlhub.vision.denoising.ffdnet import build_ffdnet_denoiser

        return build_ffdnet_denoiser(
            in_channels=in_channels, sigma=float(cfg.sigma), variant=variant
        )

    if arch in {"drunet"}:
        from dlhub.vision.denoising.drunet import build_drunet_denoiser

        return build_drunet_denoiser(
            in_channels=in_channels, sigma=float(cfg.sigma), variant=variant
        )

    if arch in {"noise2noise_unet", "n2n_unet", "noise2noise"}:
        from dlhub.vision.denoising.noise2noise import build_noise2noise_denoiser

        return build_noise2noise_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"bm3d"}:
        from dlhub.vision.denoising.bm3d import build_bm3d_denoiser

        return build_bm3d_denoiser(in_channels=in_channels, sigma=float(cfg.sigma), variant=variant)

    if arch in {"median_filter", "median"}:
        from dlhub.vision.denoising.median_filter import build_median_filter_denoiser

        return build_median_filter_denoiser(
            in_channels=in_channels, sigma=float(cfg.sigma), variant=variant
        )

    if arch in {"wiener_filter", "wiener"}:
        from dlhub.vision.denoising.wiener_filter import build_wiener_filter_denoiser

        return build_wiener_filter_denoiser(
            in_channels=in_channels, sigma=float(cfg.sigma), variant=variant
        )

    if arch in {"guided_filter", "guided"}:
        from dlhub.vision.denoising.guided_filter import build_guided_filter_denoiser

        return build_guided_filter_denoiser(
            in_channels=in_channels, sigma=float(cfg.sigma), variant=variant
        )

    if arch in {"bilateral_filter", "bilateral"}:
        from dlhub.vision.denoising.bilateral_filter import build_bilateral_filter_denoiser

        return build_bilateral_filter_denoiser(
            in_channels=in_channels, sigma=float(cfg.sigma), variant=variant
        )

    if arch in {"non_local_means", "nlm"}:
        from dlhub.vision.denoising.non_local_means import build_non_local_means_denoiser

        return build_non_local_means_denoiser(
            in_channels=in_channels, sigma=float(cfg.sigma), variant=variant
        )

    if arch in {"total_variation", "tv"}:
        from dlhub.vision.denoising.total_variation import build_total_variation_denoiser

        return build_total_variation_denoiser(
            in_channels=in_channels, sigma=float(cfg.sigma), variant=variant
        )

    if arch in {"anisotropic_diffusion", "anisodiff"}:
        from dlhub.vision.denoising.anisotropic_diffusion import (
            build_anisotropic_diffusion_denoiser,
        )

        return build_anisotropic_diffusion_denoiser(
            in_channels=in_channels, sigma=float(cfg.sigma), variant=variant
        )

    if arch in {"wavelet_shrinkage", "wavelet"}:
        from dlhub.vision.denoising.wavelet_shrinkage import build_wavelet_shrinkage_denoiser

        return build_wavelet_shrinkage_denoiser(
            in_channels=in_channels, sigma=float(cfg.sigma), variant=variant
        )

    if arch in {"anscombe_wiener", "anscombe"}:
        from dlhub.vision.denoising.anscombe_wiener import build_anscombe_wiener_denoiser

        return build_anscombe_wiener_denoiser(
            in_channels=in_channels, sigma=float(cfg.sigma), variant=variant
        )

    if arch in {"lee_filter", "lee"}:
        from dlhub.vision.denoising.lee_filter import build_lee_filter_denoiser

        return build_lee_filter_denoiser(
            in_channels=in_channels, sigma=float(cfg.sigma), variant=variant
        )

    if arch in {"kuan_filter", "kuan"}:
        from dlhub.vision.denoising.kuan_filter import build_kuan_filter_denoiser

        return build_kuan_filter_denoiser(
            in_channels=in_channels, sigma=float(cfg.sigma), variant=variant
        )

    if arch in {"stripe_remover", "stripe"}:
        from dlhub.vision.denoising.stripe_remover import build_stripe_remover_denoiser

        return build_stripe_remover_denoiser(
            in_channels=in_channels, sigma=float(cfg.sigma), variant=variant
        )

    if arch in {"dead_hot_pixel_corrector", "dead_hot_pixel", "dead_hot"}:
        from dlhub.vision.denoising.dead_hot_pixel_corrector import (
            build_dead_hot_pixel_corrector_denoiser,
        )

        return build_dead_hot_pixel_corrector_denoiser(
            in_channels=in_channels, sigma=float(cfg.sigma), variant=variant
        )

    if arch in {"line_defect_corrector", "line_defect", "stuck_line"}:
        from dlhub.vision.denoising.line_defect_corrector import (
            build_line_defect_corrector_denoiser,
        )

        return build_line_defect_corrector_denoiser(
            in_channels=in_channels, sigma=float(cfg.sigma), variant=variant
        )

    if arch in {"rowcol_bias_corrector", "rowcol_bias", "rowcol"}:
        from dlhub.vision.denoising.rowcol_bias_corrector import (
            build_rowcol_bias_corrector_denoiser,
        )

        return build_rowcol_bias_corrector_denoiser(
            in_channels=in_channels, sigma=float(cfg.sigma), variant=variant
        )

    if arch in {"block_bias_corrector", "block_bias"}:
        from dlhub.vision.denoising.block_bias_corrector import build_block_bias_corrector_denoiser

        return build_block_bias_corrector_denoiser(
            in_channels=in_channels, sigma=float(cfg.sigma), variant=variant
        )

    if arch in {"debanding_filter", "debanding", "deband"}:
        from dlhub.vision.denoising.debanding_filter import build_debanding_filter_denoiser

        return build_debanding_filter_denoiser(
            in_channels=in_channels, sigma=float(cfg.sigma), variant=variant
        )

    if arch in {"ddpm_unet", "ddpm"}:
        from dlhub.vision.denoising.ddpm_unet import build_ddpm_unet_denoiser

        return build_ddpm_unet_denoiser(
            in_channels=in_channels, sigma=float(cfg.sigma), variant=variant
        )

    if arch in {"mirnet"}:
        from dlhub.vision.denoising.mirnet import build_mirnet_denoiser

        return build_mirnet_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"mprnet"}:
        from dlhub.vision.denoising.mprnet import build_mprnet_denoiser

        return build_mprnet_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"uformer"}:
        from dlhub.vision.denoising.uformer import build_uformer_denoiser

        return build_uformer_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"cbdnet"}:
        from dlhub.vision.denoising.cbdnet import build_cbdnet_denoiser

        return build_cbdnet_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"didn"}:
        from dlhub.vision.denoising.didn import build_didn_denoiser

        return build_didn_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"rcan"}:
        from dlhub.vision.denoising.rcan import build_rcan_denoiser

        return build_rcan_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"rdn"}:
        from dlhub.vision.denoising.rdn import build_rdn_denoiser

        return build_rdn_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"memnet"}:
        from dlhub.vision.denoising.memnet import build_memnet_denoiser

        return build_memnet_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"drrn"}:
        from dlhub.vision.denoising.drrn import build_drrn_denoiser

        return build_drrn_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"rednet"}:
        from dlhub.vision.denoising.rednet import build_rednet_denoiser

        return build_rednet_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"pridnet"}:
        from dlhub.vision.denoising.pridnet import build_pridnet_denoiser

        return build_pridnet_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"dhdn"}:
        from dlhub.vision.denoising.dhdn import build_dhdn_denoiser

        return build_dhdn_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"bsn", "blindspotnet", "blind_spot_net"}:
        from dlhub.vision.denoising.bsn import build_bsn_denoiser

        return build_bsn_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"dbsn"}:
        from dlhub.vision.denoising.dbsn import build_dbsn_denoiser

        return build_dbsn_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"pixelcnn_bsn", "pcbsn"}:
        from dlhub.vision.denoising.pixelcnn_bsn import build_pixelcnn_bsn_denoiser

        return build_pixelcnn_bsn_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"gated_pixelcnn_bsn", "gpcbsn"}:
        from dlhub.vision.denoising.gated_pixelcnn_bsn import build_gated_pixelcnn_bsn_denoiser

        return build_gated_pixelcnn_bsn_denoiser(in_channels=in_channels, variant=variant)

    raise ValueError(
        f"Unknown arch: {arch_raw!r}. Examples: dncnn:dncnn_17 | restormer:restormer_tiny | bm3d:bm3d_fast"
    )


class DenoiserAdapter(nn.Module):
    """Adapter to ensure a consistent forward signature (noisy -> denoised)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model(x)
        if not isinstance(y, torch.Tensor):
            raise TypeError(f"Denoiser must return a Tensor, got: {type(y).__name__}")
        if y.shape != x.shape:
            raise ValueError(
                f"Denoiser output shape {tuple(y.shape)} must match input {tuple(x.shape)}"
            )
        return y


__all__ = ["DenoiserAdapter", "ModelConfig", "build_model", "list_supported_arches"]
