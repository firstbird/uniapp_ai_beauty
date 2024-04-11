// import { Layer, Input, Conv2D, MaxPooling2D, ZeroPadding2D, Concatenate, Model } from '@tensorflow/tfjs';
import * as tfcore from "@tensorflow/tfjs-core";
import * as tf from "@tensorflow/tfjs"
import * as tflayers from "@tensorflow/tfjs-layers";

class L2Norm extends tflayers.LayersModel {
    private axis: number;
    private gammaInit: string;
    private initialWeights: tf.Tensor;
    private nChannels: number;
    private scale: number;
    // private gamma: tf.Variable;
	private gamma: tflayers.LayerVariable | undefined;

    constructor(nChannels: number = 256, scale: number = 10, axis: number = -1, gammaInit: string = 'zero', kwargs?: any) {
        super(kwargs);
        this.axis = axis;
        this.gammaInit = gammaInit;
        this.nChannels = nChannels;
        this.scale = scale;
    }

    build(inputShape: tf.Shape): void {
        this.gamma = this.addWeight(
            'gamma',
            [this.nChannels],
            'float32',
            tf.initializers.ones()
        );

        this.built = true;
    }

    call(inputs: tf.Tensor | tf.Tensor[], kwargs?: any): tf.Tensor {
        const x = Array.isArray(inputs) ? inputs[0] : inputs;
        const norm = tf.sqrt(tf.sum(tf.square(x), this.axis, true)) as tf.Tensor;
        const scaled = tf.div(x, norm).mul(this.gamma!.read()) as tf.Tensor;
        return scaled;
    }

    getConfig(): Record<string, any> {
        const baseConfig = super.getConfig();
        return { ...baseConfig, axis: this.axis };
    }
}

export function s3fd_keras(): tf.LayersModel {
    const inp = tf.input({ shape: [null, null, 3] });

    let conv = tf.layers.conv2d({
        filters: 64,
        kernelSize: 3,
        padding: 'same',
        activation: 'relu'
    });
    let conv1_1 = conv.apply(inp) as tf.Tensor;
    let conv1_2 = conv.apply(conv1_1) as tf.Tensor;
	// todo mzl
    let maxpool1 = tf.layers.maxPooling2d({poolSize: [1, 1]}).apply(conv1_2) as tf.Tensor;

    conv = tf.layers.conv2d({
        filters: 128,
        kernelSize: 3,
        padding: 'same',
        activation: 'relu'
    });
    let conv2_1 = conv.apply(maxpool1) as tf.Tensor;
    let conv2_2 = conv.apply(conv2_1) as tf.Tensor;
    let maxpool2 = tf.layers.maxPooling2d({poolSize: [1, 1]}).apply(conv2_2) as tf.Tensor;

    conv = tf.layers.conv2d({
        filters: 256,
        kernelSize: 3,
        padding: 'same',
        activation: 'relu'
    });
    let conv3_1 = conv.apply(maxpool2) as tf.Tensor;
    let conv3_2 = conv.apply(conv3_1) as tf.Tensor;
    let conv3_3 = conv.apply(conv3_2) as tf.Tensor;
    let f3_3 = conv3_3;
    let maxpool3 = tf.layers.maxPooling2d({poolSize: [1, 1]}).apply(conv3_3) as tf.Tensor;

    conv = tf.layers.conv2d({
        filters: 512,
        kernelSize: 3,
        padding: 'same',
        activation: 'relu'
    });
    let conv4_1 = conv.apply(maxpool3) as tf.Tensor;
    let conv4_2 = conv.apply(conv4_1) as tf.Tensor;
    let conv4_3 = conv.apply(conv4_2) as tf.Tensor;
    let f4_3 = conv4_3;
    let maxpool4 = tf.layers.maxPooling2d({poolSize: [1, 1]}).apply(conv4_3) as tf.Tensor;

    conv = tf.layers.conv2d({
        filters: 512,
        kernelSize: 3,
        padding: 'same',
        activation: 'relu'
    });
    let conv5_1 = conv.apply(maxpool4) as tf.Tensor;
    let conv5_2 = conv.apply(conv5_1) as tf.Tensor;
    let conv5_3 = conv.apply(conv5_2) as tf.Tensor;
    let f5_3 = conv5_3;
    let maxpool5 = tf.layers.maxPooling2d({poolSize: [1, 1]}).apply(conv5_3) as tf.Tensor;

	// , [3, 3], [0, 0] todo mzl
    let fc6 = tf.layers.zeroPadding2d({ padding: [[0, 0], [3, 3]] }).apply(maxpool5) as tf.Tensor;
    fc6 = tf.layers.conv2d({
        filters: 1024,
        kernelSize: 3,
        activation: 'relu'
    }).apply(fc6) as tf.Tensor;
    let fc7 = tf.layers.conv2d({
        filters: 1024,
        kernelSize: 1,
        activation: 'relu'
    }).apply(fc6) as tf.Tensor;
    let ffc7 = fc7;
    let conv6_1 = tf.layers.conv2d({
        filters: 256,
        kernelSize: 1,
        activation: 'relu'
    }).apply(fc7) as tf.Tensor;
    let f6_1 = conv6_1;
    let conv6_2 = tf.layers.zeroPadding2d().apply(conv6_1) as tf.Tensor;
    conv6_2 = tf.layers.conv2d({
        filters: 512,
        kernelSize: 3,
        strides: 2,
        activation: 'relu'
    }).apply(conv6_2) as tf.Tensor;
    let f6_2 = conv6_2;
    let conv7_1 = tf.layers.conv2d({
        filters: 128,
        kernelSize: 1,
        activation: 'relu'
    }).apply(conv6_2) as tf.Tensor;
    let f7_1 = conv7_1;
    let conv7_2 = tf.layers.zeroPadding2d().apply(conv7_1) as tf.Tensor;
    conv7_2 = tf.layers.conv2d({
        filters: 256,
        kernelSize: 3,
        strides: 2,
        activation: 'relu'
    }).apply(conv7_2) as tf.Tensor;
    let f7_2 = conv7_2;

    const l2norm = (x: tf.Tensor, nChannels: number, scale: number): tf.Tensor => {
        const norm = new L2Norm(nChannels, scale).apply(x) as tf.Tensor;
        return norm;
    };

    f3_3 = l2norm(f3_3, 256, 10);
    f4_3 = l2norm(f4_3, 512, 8);
    f5_3 = l2norm(f5_3, 512, 5);

    conv = tf.layers.conv2d({
        filters: 4,
        kernelSize: 3,
        padding: 'same'
    });
    let cls1 = conv.apply(f3_3) as tf.Tensor;
    conv = tf.layers.conv2d({
        filters: 4,
        kernelSize: 3,
        padding: 'same'
    });
    let reg1 = conv.apply(f3_3) as tf.SymbolicTensor;
    conv = tf.layers.conv2d({
        filters: 2,
        kernelSize: 3,
        padding: 'same'
    });
    let cls2 = conv.apply(f4_3) as tf.SymbolicTensor;
    conv = tf.layers.conv2d({
        filters: 4,
        kernelSize: 3,
        padding: 'same'
    });
    let reg2 = conv.apply(f4_3) as tf.SymbolicTensor;
    conv = tf.layers.conv2d({
        filters: 2,
        kernelSize: 3,
        padding: 'same'
    });
    let cls3 = conv.apply(f5_3) as tf.SymbolicTensor;
    conv = tf.layers.conv2d({
        filters: 4,
        kernelSize: 3,
        padding: 'same'
    });
    let reg3 = conv.apply(f5_3) as tf.SymbolicTensor;
    conv = tf.layers.conv2d({
        filters: 2,
        kernelSize: 3,
        padding: 'same'
    });
    let cls4 = conv.apply(ffc7) as tf.SymbolicTensor;
    conv = tf.layers.conv2d({
        filters: 4,
        kernelSize: 3,
        padding: 'same'
    });
    let reg4 = conv.apply(ffc7) as tf.SymbolicTensor;
    conv = tf.layers.conv2d({
        filters: 2,
        kernelSize: 3,
        padding: 'same'
    });
    let cls5 = conv.apply(f6_2) as tf.SymbolicTensor;
    conv = tf.layers.conv2d({
        filters: 4,
        kernelSize: 3,
        padding: 'same'
    });
    let reg5 = conv.apply(f6_2) as tf.SymbolicTensor;
    conv = tf.layers.conv2d({
        filters: 2,
        kernelSize: 3,
        padding: 'same'
    });
    let cls6 = conv.apply(f7_2) as tf.SymbolicTensor;
    conv = tf.layers.conv2d({
        filters: 4,
        kernelSize: 3,
        padding: 'same'
    });
    let reg6 = conv.apply(f7_2) as tf.SymbolicTensor;

    const chunkLayer = (x: tf.Tensor, c: number): tf.Tensor[] => {
        return tf.split(x, c, -1);
    };

    //const chunk = new Lambda(chunkLayer).apply(cls1) as tf.Tensor[];
	const chunk = chunkLayer(cls1, 4);
    //const bmax = new Lambda(() => tf.maximum(tf.maximum(chunk[0], chunk[1]), chunk[2])).apply(chunk) as tf.Tensor;
    const bmax = tf.maximum(tf.maximum(chunk[0], chunk[1]), chunk[2]) as tf.Tensor;
	const cls1_out = tf.layers.concatenate().apply([bmax, chunk[3]]) as tf.SymbolicTensor;

    // todo mzl
    const model = tf.model({inputs: inp, outputs: [cls1_out, reg1, cls2, reg2, cls3, reg3, cls4, reg4, cls5, reg5, cls6, reg6]});
	return model;
	
}
