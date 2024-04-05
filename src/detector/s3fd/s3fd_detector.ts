import * as np from 'numjs';
import * as scipy from 'scipy';
import * as tflayers from "@tensorflow/tfjs-layers";
import * as tf from "@tensorflow/tfjs"

import { s3fd_keras } from './model';

export class S3FD {
    private net: tf.LayersModel;

    constructor(weightsPath: string = "./detector/s3fd/s3fd_keras_weights.h5") {
        this.net = s3fd_keras();
        // this.net.loadWeights(weightsPath);
    }

    detectFace(image: np.NDArray): number[][] {
		tf.loadLayersModel("./detector/s3fd/tfjs_model/model.json");
		 
        const bboxlist = this.detect(this.net, image);
        const keep = this.nms(bboxlist, 0.3);
        const filteredBboxlist = bboxlist.filter(x => x[4] > 0.5);
        return filteredBboxlist;
    }

    private detect(net: tf.LayersModel, img: np.NDArray): number[][] {
        const softmax = (x: np.NDArray, axis: number = -1): np.NDArray => {
            return np.exp(x.subtract(scipy.special.logsumexp(x, axis, true)));
        };

        img = img.subtract([104, 117, 123]);

        if (img.ndim === 3) {
            img = img.reshape([1, ...img.shape]);
        } else if (img.ndim === 5) {
            img = np.squeeze(img);
        }

        const [BB, HH, WW, CC] = img.shape;
        const olist = net.predict(img);

        const bboxlist: number[][] = [];
		// todo mzl
        for (let i = 0; i < olist.toString(false).length / 2; i++) {
            olist[i * 2] = softmax(olist[i * 2], -1);
        }

        for (let i = 0; i < olist.toString(false).length / 2; i++) {
            const ocls = olist[i * 2];
            const oreg = olist[i * 2 + 1];

            const [FB, FH, FW, FC] = ocls.shape;
            const stride = 2 ** (i + 2);
            const anchor = stride * 4;
            const poss = np.where(ocls.get(0, ':', ':', 1).greater(0.05));
            
            for (let i = 0; i < poss.length; i++) {
                const [Iindex, hindex, windex] = poss[i];
                const axc = stride / 2 + windex * stride;
                const ayc = stride / 2 + hindex * stride;

                const score = ocls.get(0, hindex, windex, 1);
                const loc = oreg.get(0, hindex, windex, ':');

                const priors = np.array([[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]]);
                const variances = [0.1, 0.2];
                const box = this.decode(loc, priors, variances);

                const [x1, y1, x2, y2] = box.map(v => v * 1.0);
                bboxlist.push([x1, y1, x2, y2, score]);
            }
        }

        if (bboxlist.length === 0) {
            bboxlist.push([0, 0, 0, 0, 0]);
        }

        return bboxlist;
    }

    private decode(loc: np.NDArray, priors: np.NDArray, variances: number[]): number[] {
        const boxes = np.concatenate([
            priors.get(':', ':2').add(loc.get(':', ':2').multiply(variances[0]).multiply(priors.get(':', '2:'))),
            priors.get(':', '2:').multiply(loc.get(':', '2:').exp().multiply(variances[1]))
        ], 1);

        const x1y1 = boxes.get(':', ':2').subtract(boxes.get(':', '2:').divide(2));
        const x2y2 = boxes.get(':', ':2').add(boxes.get(':', '2:').divide(2));

        return [...x1y1, ...x2y2];
    }

    private nms(dets: number[][], thresh: number): number[] {
        if (dets.length === 0) {
            return [];
        }

        const [x1, y1, x2, y2, scores] = np.transpose(dets);
        const areas = (x2.subtract(x1).add(1)).multiply(y2.subtract(y1).add(1));
        const order = scores.argsort().reverse();

        const keep: number[] = [];
        while (order.size > 0) {
            const i = order.get(0);
            keep.push(i);
            const xx1 = np.maximum(x1.get(i), x1.get(order.slice(1)));
            const yy1 = np.maximum(y1.get(i), y1.get(order.slice(1)));
            const xx2 = np.minimum(x2.get(i), x2.get(order.slice(1)));
            const yy2 = np.minimum(y2.get(i), y2.get(order.slice(1)));

            const w = np.maximum(0, xx2.subtract(xx1).add(1));
            const h = np.maximum(0, yy2.subtract(yy1).add(1));
            const ovr = w.multiply(h).divide(areas.get(i).add(areas.get(order.slice(1))).subtract(w.multiply(h)));

            const inds = np.where(ovr.lessEqual(thresh))[0];
            order.assign(order.get(inds).slice(1));
        }

        return keep;
    }
}