import cv2 from '@/uni_modules/zj-opencv';
import { Mat, MatVector} from '@/uni_modules/zj-opencv/js_sdk/opencv';


import { S3FD } from './s3fd/s3fd_detector';
import { FANLandmarksDetector } from './landmarks_detector';

abstract class BaseFaceDetector {
    constructor() {}

    abstract detectFace(image: Mat): any[];

    batchDetectFace(image: Mat): void {
        throw new Error("Method not implemented.");
    }
}

class S3FaceDetector extends BaseFaceDetector {
    private faceDetector: S3FD;

    constructor(weightsPath: string = "./models/detector/s3fd/s3fd_keras_weights.h5") {
        super();
        this.faceDetector = new S3FD(weightsPath);
    }

    detectFace(image: Mat): any[] {
        // Output bbox coordinate: y0 (left), x0 (top), y1 (right), x1 (bottom)
        return this.faceDetector.detectFace(image);
    }

    batchDetectFace(image: Mat): void {
        // throw new Error("Method not implemented.");
    }
}

export class FaceAlignmentDetector extends BaseFaceDetector {
    private fdType: string;
    private fd: S3FaceDetector | null;
    private lmdWeightsPath: string;
    private lmd: FANLandmarksDetector | null;

    constructor(fdWeightsPath: string = "./models/detector/s3fd/s3fd_keras_weights.h5",
                lmdWeightsPath: string = "./models/detector/FAN/2DFAN-4_keras.h5",
                fdType: string = "s3fd") {
        super();
        this.fdType = fdType.toLowerCase();
        this.fd = null;

        if (this.fdType === "s3fd") {
            this.fd = new S3FaceDetector(fdWeightsPath);
        } else if (this.fdType === "mtcnn") {
            throw new Error("Method not implemented.");
        } else {
            throw new Error(`Unknown face detector ${this.fdType}.`);
        }

        this.lmdWeightsPath = lmdWeightsPath;
        this.lmd = null;
    }

    buildFAN(): void {
        this.lmd = new FANLandmarksDetector(this.lmdWeightsPath);
    }

    detectFace(image: Mat, withLandmarks: boolean = true): any[] {
        let bboxList: any[] = [];

        if (this.fdType === "s3fd") {
            bboxList = this.fd!.detectFace(image);
        }

        if (bboxList.length === 0) {
            return [[], []];
        }

        let landmarksList: any[] = [];

        if (withLandmarks) {
            if (!this.lmd) {
                console.log("Building FAN for landmarks detection...");
                this.buildFAN();
                console.log("Done.");
            }

            for (const bbox of bboxList) {
                const pnts: any[] = this.lmd!.detectLandmarks(image, bbox);
                landmarksList.push([pnts]);
            }

            landmarksList = landmarksList.map(landmarks => this.postProcessLandmarks(landmarks));
            bboxList = this.preprocessS3FDBbox(bboxList);

            return [bboxList, landmarksList];
        } else {
            bboxList = this.preprocessS3FDBbox(bboxList);
            return [bboxList];
        }
    }

    private batchDetectFace(images: Mat[], kwargs: any) : void {
    //     throw new Error("Method not implemented.");
    }

    private preprocessS3FDBbox(bboxList: any[]): any[][] {
        // Convert coord (y0, x0, y1, x1) to (x0, y0, x1, y1)
        return bboxList.map(bbox => [bbox[1], bbox[0], bbox[3], bbox[2], bbox[4]]);
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

    private postProcessLandmarks(landmarks: any[]): any[] {
        // Process landmarks to have shape [68, 2]
		const newShape: number[] = [68, 2];
        const lms: any[][] = landmarks.map(landmark => this.reshapeArray([landmark], newShape).map(pnt => pnt.reverse()));
        return lms;
    }

    static drawLandmarks(image: Mat, landmarks: any[], color: any = new cv2.Vec(0, 255, 0), stroke: number = 3): Mat {
        for (const landmark of landmarks) {
            const [x, y] = landmark;
            cv2.circle(image.clone(), new cv2.Point(y, x), stroke, color, -1);
        }
        return image;
    }
}
