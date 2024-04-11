// import { Conv2D, BatchNormalization, Activation, MaxPooling2D, Add, GlobalAveragePooling2D, Reshape, Lambda, Multiply, Concatenate } from '@tensorflow/tfjs';
import * as tfcore from "@tensorflow/tfjs-core";
import * as tf from "@tensorflow/tfjs";
import * as tflayers from "@tensorflow/tfjs-layers";



function conv_block(x: tf.SymbolicTensor, f: number, k: number, block_name: string = "", layer_id: string = "", s: number = 1,  use_activ: boolean = true): tf.SymbolicTensor {
    if (k !== 1) {
        x = tf.layers.zeroPadding2d({ padding: [[1, 1], [1, 1]] }).apply(x) as tf.SymbolicTensor;
    }
    x = tf.layers.conv2d({
        filters: f,
        kernelSize: k,
        strides: s,
        padding: 'valid',
        useBias: false,
        name: `${block_name}.conv${layer_id}`
    }).apply(x) as tf.SymbolicTensor;
    x = tf.layers.batchNormalization({ epsilon: 1e-5, name: `${block_name}.bn${layer_id}` }).apply(x) as tf.SymbolicTensor;
    x = use_activ ? tf.layers.activation({ activation: 'relu' }).apply(x) as tf.SymbolicTensor : x;
    return x;
}

function res_block_with_downsampling(x: tf.SymbolicTensor, f: number, block_name: string = "cp.resnet.layerN"): tf.SymbolicTensor {
    let skip = tf.layers.conv2d({
        filters: f,
        kernelSize: 1,
        strides: 2,
        useBias: false,
        name: `${block_name}.0.downsample.0`
    }).apply(x) as tf.SymbolicTensor;
    skip = tf.layers.batchNormalization({ epsilon: 1e-5, name: `${block_name}.0.downsample.1` }).apply(skip) as tf.SymbolicTensor;
    x = conv_block(x, f, 3, `${block_name}.0`, "1", 2);
    x = conv_block(x, f, 3, `${block_name}.0`, "2", 1,  false);
    x = tf.layers.add().apply([x, skip]) as tf.SymbolicTensor;
    x = tf.layers.activation({ activation: 'relu' }).apply(x) as tf.SymbolicTensor;

    skip = x;
    x = conv_block(x, f, 3, `${block_name}.1`, "1", 1);
    x = conv_block(x, f, 3, `${block_name}.1`, "2", 1, false);
    x = tf.layers.add().apply([x, skip]) as tf.SymbolicTensor;
    x = tf.layers.activation({ activation: 'relu' }).apply(x) as tf.SymbolicTensor;
    return x;
}

function attention_refinment_block(x: tf.SymbolicTensor, f: number, block_name: string = "cp.arm16"): tf.SymbolicTensor {
    x = tf.layers.conv2d({
        filters: f,
        kernelSize: 3,
        padding: 'same',
        useBias: false,
        name: `${block_name}.conv.conv`
    }).apply(x) as tf.SymbolicTensor;
    x = tf.layers.batchNormalization({ epsilon: 1e-5, name: `${block_name}.conv.bn` }).apply(x) as tf.SymbolicTensor;
    x = tf.layers.activation({ activation: 'relu' }).apply(x) as tf.SymbolicTensor;

	// todo mzl
    let attn = tf.layers.globalAveragePooling2d({}).apply(x) as tf.SymbolicTensor;
    attn = tf.layers.reshape({ targetShape: [1, 1, f] }).apply(attn) as tf.SymbolicTensor;
    attn = tf.layers.conv2d({
        filters: f,
        kernelSize: 1,
        useBias: false,
        name: `${block_name}.conv_atten`
    }).apply(attn) as tf.SymbolicTensor;
    attn = tf.layers.batchNormalization({ epsilon: 1e-5, name: `${block_name}.bn_atten` }).apply(attn) as tf.SymbolicTensor;
    attn = tf.layers.activation({ activation: 'sigmoid' }).apply(attn) as tf.SymbolicTensor;
    x = tf.layers.multiply().apply([x, attn]) as tf.SymbolicTensor;
    return x;
}

function feature_fusion_block(x1: tf.SymbolicTensor, x2: tf.SymbolicTensor): tf.SymbolicTensor {
    let x = tf.layers.concatenate().apply([x1, x2]) as tf.SymbolicTensor;
    x = conv_block(x, 256, 1, "ffm.convblk", "");
	// todo mzl
    let attn = tf.layers.globalAveragePooling2d({}).apply(x) as tf.SymbolicTensor;
    attn = tf.layers.reshape({ targetShape: [1, 1, 256] }).apply(attn) as tf.SymbolicTensor;
    attn = tf.layers.conv2d({ filters: 64, kernelSize: 1, useBias: false, name: "ffm.conv1" }).apply(attn) as tf.SymbolicTensor;
    attn = tf.layers.activation({ activation: 'relu' }).apply(attn) as tf.SymbolicTensor;
    attn = tf.layers.conv2d({ filters: 256, kernelSize: 1, useBias: false, name: "ffm.conv2" }).apply(attn) as tf.SymbolicTensor;
    let feat_attn = tf.layers.activation({ activation: 'sigmoid' }).apply(attn) as tf.SymbolicTensor;
    attn = tf.layers.multiply().apply([x, feat_attn]) as tf.SymbolicTensor;    
    x = tf.layers.add().apply([x, attn]) as tf.SymbolicTensor;
    return x;
}

function upsampling(x: tf.SymbolicTensor, shape: number[], interpolation: string = "nearest"): tf.SymbolicTensor {
    if (interpolation === "nearest") {
		// todo mzl
        // return tf.layers.lambda((t) => tf.image.resizeNearestNeighbor(t, shape)).apply(x) as tf.SymbolicTensor;
		// return tf.layers.CustomCallback.bind((t) => tf.image.resizeNearestNeighbor(t, [shape[0], shape[1]])).apply(x) as tf.SymbolicTensor;
    
	} else if (interpolation === "bilinear") {
        // return tflayers.CustomCallback.bind((t) => tf.image.resizeBilinear(t, [shape[0], shape[1]])).apply(x) as tf.SymbolicTensor;
    }
    return  x = tf.layers.maxPooling2d({ poolSize: [100, 100], strides: [4, 4] }).apply(x) as tf.SymbolicTensor;
;
}

function maxpool(x: tf.SymbolicTensor, k: number = 3, s: number = 2, pad: number = 1): tf.SymbolicTensor {
    x = tf.layers.zeroPadding2d({ padding: [[pad, pad], [pad, pad]] }).apply(x) as tf.SymbolicTensor;
    x = tf.layers.maxPooling2d({ poolSize: [k, k], strides: [s, s] }).apply(x) as tf.SymbolicTensor;
    return x;
}

export function BiSeNet_keras(input_resolution: number = 512): tf.LayersModel {
    const inp = tf.input({ shape: [input_resolution, input_resolution, 3] });
    let x = tf.layers.zeroPadding2d({ padding: [[3, 3], [3, 3]] }).apply(inp) as tf.SymbolicTensor;
    x = tf.layers.conv2d({ filters: 64, kernelSize: 7, strides: 2, useBias: false, name: "cp.resnet.conv1" }).apply(x) as tf.SymbolicTensor;
    x = tf.layers.batchNormalization({ epsilon: 1e-5, name: "cp.resnet.bn1" }).apply(x) as tf.SymbolicTensor;
    x = tf.layers.activation({ activation: 'relu' }).apply(x) as tf.SymbolicTensor;
    x = maxpool(x);

    // layer1
    let skip = x;
    x = conv_block(x, 64, 3, "cp.resnet.layer1.0", "1", 1);
    x = conv_block(x, 64, 3, "cp.resnet.layer1.0", "2", 1, false);
    x = tf.layers.add().apply([x, skip]) as tf.SymbolicTensor;
    x = tf.layers.activation({ activation: 'relu' }).apply(x) as tf.SymbolicTensor;
    skip = x;
    x = conv_block(x, 64, 3, "cp.resnet.layer1.1", "1", 1);
    x = conv_block(x, 64, 3, "cp.resnet.layer1.1", "2", 1, false);
    x = tf.layers.add().apply([x, skip]) as tf.SymbolicTensor;
    x = tf.layers.activation({ activation: 'relu' }).apply(x) as tf.SymbolicTensor;

    // layer2
    x = res_block_with_downsampling(x, 128, "cp.resnet.layer2");
    let feat8 = x;

    // layer3
    x = res_block_with_downsampling(x, 256, "cp.resnet.layer3");
    let feat16 = x;

    // ARM1
    let feat16_arm = attention_refinment_block(feat16, 128, "cp.arm16");

    // layer4
    x = res_block_with_downsampling(x, 512, "cp.resnet.layer4");
    let feat32 = x;

    // ARM2 and conv_avg
    let conv_avg = tf.layers.globalAveragePooling2d({}).apply(x) as tf.SymbolicTensor;
    conv_avg = tf.layers.reshape({ targetShape: [1, 1, 512] }).apply(conv_avg) as tf.SymbolicTensor;
    conv_avg = conv_block(conv_avg, 128, 1, "cp.conv_avg", "");
    let avg_up = upsampling(conv_avg, [input_resolution / 32, input_resolution / 32]);
    let feat32_arm = attention_refinment_block(x, 128, "cp.arm32");
    let feat32_sum = tf.layers.add().apply([feat32_arm, avg_up]) as tf.SymbolicTensor;
    let feat32_up = upsampling(feat32_sum, [input_resolution / 16, input_resolution / 16]);
    feat32_up = conv_block(feat32_up, 128, 3, "cp.conv_head32", "");

    let feat16_sum = tf.layers.add().apply([feat16_arm, feat32_up]) as tf.SymbolicTensor;
    let feat16_up = upsampling(feat16_sum, [input_resolution / 8, input_resolution / 8]);
    feat16_up = conv_block(feat16_up, 128, 3, "cp.conv_head16", "");

    // FFM
    let feat_sp = feat8;
    let feat_cp8 = feat16_up;
    let feat_fuse = feature_fusion_block(feat_sp, feat_cp8);

    let feat_out = conv_block(feat_fuse, 256, 3, "conv_out.conv", "");
    feat_out = tf.layers.conv2d({ filters: 19, kernelSize: 1, strides: 1, useBias: false, name: "conv_out.conv_out" }).apply(feat_out) as tf.SymbolicTensor;
    feat_out = upsampling(feat_out, [input_resolution, input_resolution], "bilinear");
    // Ignore feat_out32 and feat_out16 since they are not used in inference phase

    return tf.model({ inputs: inp, outputs: feat_out });
}
