import * as cv2 from '@/uni_modules/zj-opencv';
import * as np from 'numjs';
import * as tf from "@tensorflow/tfjs";


import { BiSeNet_keras } from './BiSeNet/bisenet';

export class FaceParser {
    private parserNet: tf.LayersModel | null;
    private detector: any;

    constructor(pathBiSeNetWeights: string = "./models/parser/BiSeNet/BiSeNet_keras.h5", detector: any = null) {
        // this.parserNet = null;
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
                    let [y0, x0, y1, x1, _] = bbox;
                    x0 = Math.max(x0, 0);
                    y0 = Math.max(y0, 0);
                    x1 = Math.min(x1, origH);
                    y1 = Math.min(y1, origW);
                    x0 = Math.floor(x0);
                    y0 = Math.floor(y0);
                    x1 = Math.floor(x1);
                    y1 = Math.floor(y1);
                    return im.getRegion(new cv2.Rect(x0, y0, x1 - x0, y1 - y0));
                });
            } else {
                faces = [im];
            }
        } else {
            const [x0, y0, x1, y1] = boundingBox;
            const startX: number = Math.max(x0, 0);
            const startY: number = Math.max(y0, 0);
            const endX: number = Math.min(x1, origH);
            const endY: number = Math.min(y1, origW);
            const faceRegion: any = im.getRegion(new cv2.Rect(startX, startY, endX - startX, endY - startY));
            faces = [faceRegion];
        }

        let maps: any[] = [];
        for (const face of faces) {
            const [origH, origW] = face.sizes;
            let inp: any = face.resize(new cv2.Size(512, 512));
            inp = this.normalizeInput(inp);
            inp = inp.expandDims(0);

            const out: np.NDArray = this.parserNet.predict([inp])[0];
            let parsingMap: np.NDArray = out.argmax(-1);
            parsingMap = parsingMap.resize(new cv2.Size(origW, origH), 0, 0, cv2.InterpolationFlags.INTER_NEAREST);
            maps.push(parsingMap);
        }
        return maps;
    }

    private normalizeInput(x: any, mean: number[] = [0.485, 0.456, 0.406], std: number[] = [0.229, 0.224, 0.225]): any {
        return x.div(255).sub(mean).div(std);
    }
}