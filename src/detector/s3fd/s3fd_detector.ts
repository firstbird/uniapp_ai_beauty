// import * as scipy from 'scipy';
import * as tflayers from "@tensorflow/tfjs-layers";
import * as tf from "@tensorflow/tfjs"
import { Mat, MatVector} from '@/uni_modules/zj-opencv/js_sdk/opencv';
import cv2 from '@/uni_modules/zj-opencv';



import { s3fd_keras } from './model';

export class S3FD {
    private net: tf.LayersModel;

    constructor(weightsPath: string = "./detector/s3fd/s3fd_keras_weights.h5") {
        this.net = s3fd_keras();
        // this.net.loadWeights(weightsPath);
    }

    detectFace(image: Mat): number[][] {
		tf.loadLayersModel("./detector/s3fd/tfjs_model/model.json");
		 
        const bboxlist = this.detect(this.net, image);
        const keep = this.nms(bboxlist, 0.3);
        const filteredBboxlist = bboxlist.filter(x => x[4] > 0.5);
        return filteredBboxlist;
    }

	// private logsumexp(x: number[], axis: number | null = null, keepdims: boolean = false): number {
	  // if (axis === null) {
		// const maxVal = Math.max(...x);
		// const sumExp = x.map((xi) => Math.exp(xi - maxVal)).reduce((acc, val) => acc + val, 0);
		// if (keepdims) {
		//   return Math.log(sumExp) + maxVal;
		// } else {
		//   return Math.log(sumExp);
		// }
	 //  } else {
		// const maxVals = x.map((row) => Math.max(...row));
		// const exps = x.map((row, rowIndex) => row.map((val, colIndex) => Math.exp(val - maxVals[rowIndex])));
		// const sums = exps.map((row) => row.reduce((acc, val) => acc + val, 0));
		// if (keepdims) {
		//   return sums.map((val, index) => Math.log(val) + maxVals[index]);
		// } else {
		//   return sums.map((val) => Math.log(val));
		// }
	 //  }
	// }
	
	logsumexp(x: number[], axis: number, keepdims: boolean): number {
	  if (keepdims) {
	    const maxVal = Math.max(...x);
	    const sumExp = x.map((xi) => Math.exp(xi - maxVal)).reduce((acc, val) => acc + val, 0);
	    return maxVal + Math.log(sumExp);
	  } else {
	    const maxVal = Math.max(...x);
	    const sumExp = x.map((xi) => Math.exp(xi - maxVal)).reduce((acc, val) => acc + val, 0);
	    return Math.log(sumExp) + maxVal;
	  }
	}
	
	private reshapeArray(inputArray: number[], shape: number[]): number[][] {
	  if (inputArray.length !== shape.reduce((a, b) => a * b, 1)) {
		throw new Error('Total size of new array must be unchanged');
	  }

	  let result: number[][] = [];
	  let index = 0;
	  for (let i = 0; i < shape[0]; i++) {
		let row: number[] = [];
		for (let j = 0; j < shape[1]; j++) {
		  row.push(inputArray[index]);
		  index++;
		}
		result.push(row);
	  }
	  return result;
	}
	
    private detect(net: tf.LayersModel, img: Mat): number[][] {
        const softmax = (x: number[], axis: number = -1): number[] => {
			const logsumexpResult = this.logsumexp(x, axis, true);
			const expResult = x.map(function(xi) {
			    return Math.exp(xi - logsumexpResult);
			  });
			return expResult;
            // return np.exp(x.subtract(this.logsumexp(x, axis, true)));
        };

		let imgout = new cv2.Mat();
		// todo mzl
		let minus = cv2.matFromArray(1, 3, cv2.CV_8UC1, [104, 117, 123]);

        cv2.subtract(img, minus, imgout);

        if (img.rows === 3) {
			cv2.resize(imgout, img, new cv2.Size(1, img.rows * img.cols));
            // img = img.reshape([1, ...img.shape]);
        } else if (img.rows === 5) {
			// todo mzl
            // img = np.squeeze(img);
        }

        // const [BB, HH, WW, CC] = img.shape;// [batch_size, height, width, channels]
		// olist is Mat
        let olist = net.predict(tf.tensor(img.data, [img.rows, img.cols, -1]));

        const bboxlist: number[][] = [];
		// todo mzl
		const olist_arr: number[][] = JSON.parse(olist.toString());
        for (let i = 0; i < olist_arr.length / 2; i++) {
            olist_arr[i * 2] = softmax(olist_arr[i * 2], -1);
        }

        for (let i = 0; i < olist_arr.length / 2; i++) {
            const ocls = olist_arr[i * 2];
            const oreg = olist_arr[i * 2 + 1];

            const [FB, FH, FW, FC] = ocls;
            const stride = 2 ** (i + 2);
            const anchor = stride * 4;
            //const poss = np.where(ocls.get(0, ':', ':', 1).greater(0.05));
			let poss: [number, number][] = [];

			// for (let i = 0; i < ocls.length; i++) {
			// 	for (let j = 0; j < ocls[i].length; j++) {
			// 		if (ocls[j] > 0.05) {
			// 			poss.push([i, j]);
			// 		}
			// 	}
			// }
    //         for (let i = 0; i < poss.length; i++) {
				// const windex = i;
    //             const [Iindex, hindex] = poss[i];
    //             const axc = stride / 2 + windex * stride;
    //             const ayc = stride / 2 + hindex * stride;
				// // todo mzl 
    //             const score = ocls[Iindex, windex];
    //             const loc = [oreg[Iindex, windex]];

    //             const priors = [[[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]]];
    //             const variances = [0.1, 0.2];
    //             const box = this.decode(loc, priors, variances);

    //             const [x1, y1, x2, y2] = box.map(v => v * 1.0);
    //             bboxlist.push([x1, y1, x2, y2, score]);
    //         }
        }

        if (bboxlist.length === 0) {
            bboxlist.push([0, 0, 0, 0, 0]);
        }

        return bboxlist;
    }

    private decode(loc: number[], priors: number[], variances: number[]): number[] {
        // const boxes = np.concatenate([
        //     priors.get(':', ':2').add(loc.get(':', ':2').multiply(variances[0]).multiply(priors.get(':', '2:'))),
        //     priors.get(':', '2:').multiply(loc.get(':', '2:').exp().multiply(variances[1]))
        // ], 1);
		// todo mzl

        // const x1y1 = boxes.get(':', ':2').subtract(boxes.get(':', '2:').divide(2));
        // const x2y2 = boxes.get(':', ':2').add(boxes.get(':', '2:').divide(2));

        // return [...x1y1, ...x2y2];
		return [];
    }
	private transposeMatrix(matrix: number[][]): number[][] {
		const rows = matrix.length;
		const cols = matrix[0].length;

		let result: number[][] = [];
		for (let j = 0; j < cols; j++) {
			result[j] = [];
			for (let i = 0; i < rows; i++) {
				result[j][i] = matrix[i][j];
			}
		}

		return result;
	}
    private nms(dets: number[][], thresh: number): number[] {
        if (dets.length === 0) {
            return [];
        }

        const [x1, y1, x2, y2, scores] = this.transposeMatrix(dets);
		// todo mzl
        // const areas = (x2.subtract(x1).add(1)).multiply(y2.subtract(y1).add(1));
        // const order = scores.argsort().reverse();

        const keep: number[] = [];
        // while (order.size > 0) {
        //     const i = order.get(0);
        //     keep.push(i);
        //     const xx1 = np.maximum(x1.get(i), x1.get(order.slice(1)));
        //     const yy1 = np.maximum(y1.get(i), y1.get(order.slice(1)));
        //     const xx2 = np.minimum(x2.get(i), x2.get(order.slice(1)));
        //     const yy2 = np.minimum(y2.get(i), y2.get(order.slice(1)));

        //     const w = np.maximum(0, xx2.subtract(xx1).add(1));
        //     const h = np.maximum(0, yy2.subtract(yy1).add(1));
        //     const ovr = w.multiply(h).divide(areas.get(i).add(areas.get(order.slice(1))).subtract(w.multiply(h)));

        //     const inds = np.where(ovr.lessEqual(thresh))[0];
        //     order.assign(order.get(inds).slice(1));
        // }

        return keep;
    }
}