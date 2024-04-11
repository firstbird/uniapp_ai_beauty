// import * as np from 'numjs';
import * as tf from "@tensorflow/tfjs";
import cv2 from '@/uni_modules/zj-opencv';
import { Mat, MatVector} from '@/uni_modules/zj-opencv/js_sdk/opencv';



import { BiSeNet_keras } from './BiSeNet/bisenet';

export class FaceParser {
    private parserNet: tf.LayersModel | null;
    private detector: any;

    constructor(pathBiSeNetWeights: string = "./models/parser/BiSeNet/BiSeNet_keras.h5", detector: any = null) {
        this.parserNet = null;
        this.detector = detector;

        this.buildParserNet(pathBiSeNetWeights);
    }

    private async buildParserNet(path: string) {
        const parserNet = BiSeNet_keras();
	    await tf.loadLayersModel("./models/parser/BiSeNet/tfjs_model/model.json");
        // parserNet.loadWeights(path);
		
        this.parserNet = parserNet;
    }

    public setDetector(detector: any): void {
        this.detector = detector;
    }

    public removeDetector(): void {
        this.detector = null;
    }

    public parseFace(im: any, boundingBox: number[] | null = null, withDetection: boolean = false): any[] {
        const origH: number = im.rows;
        const origW: number = im.cols;

        let faces: any[] = [];

        // Detect/Crop face RoI
        if (boundingBox === null) {
            if (withDetection) {
                if (!this.detector.fd) {
                    throw new Error("Error occurs during face detection: detector not found in FaceParser.");
                }

                const bboxes: any[] = this.detector.fd.detectFace(im);
                faces = bboxes.map(bbox => {
                    const [y0, x0, y1, x1] = bbox;
                    const [x0_clipped, y0_clipped] = [Math.max(x0, 0), Math.max(y0, 0)];
                    const [x1_clipped, y1_clipped] = [Math.min(x1, origH), Math.min(y1, origW)];
                    const [x0_int, y0_int, x1_int, y1_int] = [Math.floor(x0_clipped), Math.floor(y0_clipped), Math.floor(x1_clipped), Math.floor(y1_clipped)];
                    // todo mzl confirm
                    //return [im.slice(x0_int, x1_int, y0_int, y1_int)];
                    return [im.colRange(x0_int, x1_int).rowRange(y0_int, y1_int)];
                });
            } else {
                faces = [im];
            }
        } else {
            const [x0, y0, x1, y1] = boundingBox;
            const [x0_clipped, y0_clipped] = [Math.max(x0, 0), Math.max(y0, 0)];
            const [x1_clipped, y1_clipped] = [Math.min(x1, origH), Math.min(y1, origH)];
            const [x0_int, y0_int, x1_int, y1_int] = [Math.floor(x0_clipped), Math.floor(y0_clipped), Math.floor(x1_clipped), Math.floor(y1_clipped)];
            faces = [im.colRange(x0_int, x1_int).rowRange(y0_int, y1_int)];
        }

        let maps: any[] = [];
        for (const face of faces) {
            const [origH, origW] = face.sizes;
            let inp: any = face.resize(new cv2.Size(512, 512));
            inp = this.normalizeInput(inp);
            // todo mzl
            //inp = inp.reshape([1, ...inp.shape]);

            // todo mzl
            // const out = this.parser_net.predict([inp])[0];
            // let parsing_map = out.argmax(-1);
            let parsing_map : Mat = new cv2.Mat();
            let resize_map : Mat = new cv2.Mat();
            // parsing_map(np.uint8)
            cv2.resize(parsing_map, resize_map, new cv2.Size(origW, origH)); // , cv2.INTER_NEAREST
            maps.push(parsing_map);
        }
        return maps;
    }

    private normalizeInput(x: Mat, mean: number[] = [0.485, 0.456, 0.406], std: number[] = [0.229, 0.224, 0.225]): Mat {
        //return (- mean) / std;
        // todo mzl
        return (x.mul(cv2.Mat.eye(x.cols, x.rows, cv2.CV_8UC1), 1.0 / 255.0));
    }
}