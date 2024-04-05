import cv2 from '@/uni_modules/zj-opencv';

import * as np from 'numjs';

// import { loadModel } from '@tensorflow/tfjs-node';
import * as tf from "@tensorflow/tfjs";

abstract class BaseLandmarksDetector {
    constructor() {
        throw new Error("Method not implemented.");
    }

    abstract detectLandmarks(image: cv2.Mat, boundingBox?: any, faceDetector?: any): any[];
}

export class FANLandmarksDetector extends BaseLandmarksDetector {
    private net: any;

    constructor(pathToWeightsFile: string = "./models/detector/FAN/2DFAN-4_keras.h5") {
        super();
        if (process.env.TFJS_NODE_VERSION && parseInt(process.env.TFJS_NODE_VERSION.split('.')[0]) >= 1 && parseInt(process.env.TFJS_NODE_VERSION.split('.')[1]) >= 13) {
            // this.net = await loadModel(`file://${pathToWeightsFile}`);
			this.run();
        } else {
            throw new Error("TensorFlow version must be at least 1.13.");
        }
    }
	
	async run() : Promise<void>
	{
		this.net = await tf.loadLayersModel("./detector/FAN/tfjs_model/model.json");
	}

    detectLandmarks(image: cv2.Mat, boundingBox: any = null, faceDetector: any = null): any[] {
        if (!boundingBox && !faceDetector) {
            throw new Error("Neither bounding box or face detector is passed in.");
        }
		

        const detect: boolean = !boundingBox;

        let prepImg: np.ndarray;
        let center: np.ndarray;
        let scale: number;

        [prepImg, center, scale] = this._preprocessingFAN(image, detect, faceDetector, boundingBox);

        const pred: any = this.net.predict(prepImg.expandDims(0));

        const [pnts, pntsOrig]: [any[], any[]] = this._getPredsFromhm(pred.slice(-1), center, scale);

        return [pnts, pntsOrig];
    }

    private _preprocessingFAN(img: np.ndarray, detect: boolean = false, faceDetector: any = null, bbox: any = null): [np.ndarray, np.ndarray, number] {
        if (img.dims === 2) {
            img = np.stack([img, img, img], -1);
        } else if (img.dims === 4) {
            img = img.slice(null, null, null, 3);
        }

        let x0: number, x1: number, y0: number, y1: number;

        if (detect) {
            if (!faceDetector) {
                throw new Error(`face_detector has not been specified. face_detect is [${faceDetector}]`);
            }
            bbox = faceDetector.detectFace(img)[0];
            [x0, x1, y0, y1] = [bbox[1], bbox[3], bbox[0], bbox[2]];
        } else {
            [x0, x1, y0, y1] = [bbox[1], bbox[3], bbox[0], bbox[2]];
        }

        const center: np.ndarray = np.array([(y0 + y1) / 2, (x0 + x1) / 2], np.float32);
        center.set(1, center.get(1) - (x1 - x0) * 0.12);
        const scale: number = (x1 - x0 + y1 - y0) / 195;

        let newImg: np.ndarray = this._crop(img, center, scale);

        newImg = newImg.transpose(2, 0, 1);
        newImg = newImg.div(255);

        return [newImg, center, scale];
    }

    private _crop(image: np.ndarray, center: np.ndarray, scale: number, resolution: number = 256.0): np.ndarray {
        const ul: np.ndarray = this._transform([1, 1], center, scale, resolution, true);
        const br: np.ndarray = this._transform([resolution, resolution], center, scale, resolution, true);

        let newDim: np.ndarray;
        if (image.dims > 2) {
            newDim = np.array([br.get(1) - ul.get(1), br.get(0) - ul.get(0), image.shape.get(2)], np.int32);
            newImg = np.zeros(newDim, np.uint8);
        } else {
            newDim = np.array([br.get(1) - ul.get(1), br.get(0) - ul.get(0)], np.int);
            newImg = np.zeros(newDim, np.uint8);
        }

        const ht: number = image.shape.get(0);
        const wd: number = image.shape.get(1);
        const newX: np.ndarray = np.array([Math.max(1, -ul.get(0) + 1), Math.min(br.get(0), wd) - ul.get(0)], np.int32);
        const newY: np.ndarray = np.array([Math.max(1, -ul.get(1) + 1), Math.min(br.get(1), ht) - ul.get(1)], np.int32);
        const oldX: np.ndarray = np.array([Math.max(1, ul.get(0) + 1), Math.min(br.get(0), wd)], np.int32);
        const oldY: np.ndarray = np.array([Math.max(1, ul.get(1) + 1), Math.min(br.get(1), ht)], np.int32);
        newImg.getRegion(newY.get(0) - 1, newX.get(0) - 1, newY.get(1), newX.get(1))
            .assign(image.getRegion(oldY.get(0) - 1, oldX.get(0) - 1, oldY.get(1), oldX.get(1)));

        newImg = cv2.resize(newImg, new cv2.Size(resolution, resolution));

        return newImg;
    }

    private _transform(point: number[], center: np.ndarray, scale: number, resolution: number, invert: boolean = false): np.ndarray {
        const _pt: np.ndarray = np.ones(3);
        _pt.set(0, point[0]);
        _pt.set(1, point[1]);

        const h: number = 200.0 * scale;
        let t: np.ndarray = np.eye(3);
        t.set(0, 0, resolution / h);
        t.set(1, 1, resolution / h);
        t.set(0, 2, resolution * (-center.get(0) / h + 0.5));
        t.set(1, 2, resolution * (-center.get(1) / h + 0.5));

        if (invert) {
            t = np.linalg.inv(t);
        }

        const newPoint: np.ndarray = np.matmul(t, _pt).slice(0, 2).astype(np.int32);

        return newPoint;
    }
	
	private _getPredsFromhm(hm: np.NDArray, center: np.NDArray | null = null, scale: number | null = null): [np.NDArray[], np.NDArray[]] {
	    if (hm.ndim !== 4) {
	        throw new Error(`Received hm in unexpected dimension: ${hm.ndim}.`);
	    }
	
	    const hmFlat: np.NDArray = hm.reshape([hm.shape[1], hm.shape[2] * hm.shape[3]]);
	    const idx: np.NDArray = np.argmax(hmFlat, -1).expandDims(0);
	    idx.iadd(1);
	
	    let preds: np.NDArray = np.repeat(idx.reshape([idx.shape[0], idx.shape[1], 1]), 2, 2).astype(np.float32);
	    preds.getNDArray().mapInplace((val: number, idx: number[]) => {
	        idx[2] === 0 ? (val - 1) % hm.shape[3] + 1 : Math.floor((val - 1) / hm.shape[2]) + 1;
	    });
	
	    for (let i = 0; i < preds.shape[0]; i++) {
	        for (let j = 0; j < preds.shape[1]; j++) {
	            const hm_: np.NDArray = hm.get(i).get(j);
	            let [pX, pY]: number[] = [preds.get(i).get(j).get(0) - 1, preds.get(i).get(j).get(1) - 1];
	            if (pX > 0 && pX < 63 && pY > 0 && pY < 63) {
	                const diff: np.NDArray = np.array([
	                    hm_.get(pY, pX + 1) - hm_.get(pY, pX - 1),
	                    hm_.get(pY + 1, pX) - hm_.get(pY - 1, pX)
	                ]);
	                preds.get(i).get(j).iadd(np.sign(diff).mul(0.25));
	            }
	        }
	    }
	
	    preds = preds.sub(0.5);
	
	    const predsOrig: np.NDArray = np.zerosLike(preds);
	    if (center && scale) {
	        for (let i = 0; i < hm.shape[0]; i++) {
	            for (let j = 0; j < hm.shape[1]; j++) {
	                predsOrig.get(i).get(j).assign(this._transform(
	                    preds.get(i).get(j), center, scale, hm.shape[2], true));
	            }
	        }
	    }
	
	    return [preds, predsOrig];
	}
	
	drawLandmarks(image: np.NDArray, landmarks: [number, number][], color: [number, number, number]): np.NDArray {
	    for (let i = 0; i < landmarks.length; i++) {
	        const [x, y] = landmarks[i];
	        image = cv2.circle(image.copy(), [Math.floor(x), Math.floor(y)], 3, color, -1);
	    }
	    return image;
	}
}