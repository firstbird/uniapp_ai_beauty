import cv2 from '@/uni_modules/zj-opencv';
import * as np from 'numjs';

import { S3FD } from './s3fd/s3fd_detector';
import { FANLandmarksDetector } from './landmarks_detector';

abstract class BaseFaceDetector {
    constructor() {}

    abstract detectFace(image: cv2.Mat): any[];

    batchDetectFace(image: cv2.Mat): void {
        throw new Error("Method not implemented.");
    }
}

class S3FaceDetector extends BaseFaceDetector {
    private faceDetector: S3FD;

    constructor(weightsPath: string = "./models/detector/s3fd/s3fd_keras_weights.h5") {
        super();
        this.faceDetector = new S3FD(weightsPath);
    }

    detectFace(image: cv2.Mat): any[] {
        // Output bbox coordinate: y0 (left), x0 (top), y1 (right), x1 (bottom)
        return this.faceDetector.detectFace(image);
    }

    batchDetectFace(image: cv2.Mat): void {
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

    detectFace(image: cv2.Mat, withLandmarks: boolean = true): any[] {
        let bboxList: any[] = [];

        if (this.fdType === "s3fd") {
            bboxList = this.fd.detectFace(image);
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
                const pnts: any[] = this.lmd.detectLandmarks(image, bbox);
                landmarksList.push(np.array(pnts));
            }

            landmarksList = landmarksList.map(landmarks => this.postProcessLandmarks(landmarks));
            bboxList = this.preprocessS3FDBbox(bboxList);

            return [bboxList, landmarksList];
        } else {
            bboxList = this.preprocessS3FDBbox(bboxList);
            return [bboxList];
        }
    }

    private batchDetectFace(images: cv2.Mat[], kwargs: any) : void {
    //     throw new Error("Method not implemented.");
    }

    private preprocessS3FDBbox(bboxList: any[]): any[] {
        // Convert coord (y0, x0, y1, x1) to (x0, y0, x1, y1)
        return bboxList.map(bbox => np.array([bbox[1], bbox[0], bbox[3], bbox[2], bbox[4]]));
    }

    private postProcessLandmarks(landmarks: any[]): any[] {
        // Process landmarks to have shape [68, 2]
        const lms: any[][] = landmarks.map(landmark => np.array(landmark).reshape(68, 2).map(pnt => pnt.reverse()));
        return lms;
    }

    static drawLandmarks(image: cv2.Mat, landmarks: any[], color: any = new cv2.Vec3(0, 255, 0), stroke: number = 3): cv2.Mat {
        for (const landmark of landmarks) {
            const [x, y] = landmark;
            image = cv2.circle(image.copy(), new cv2.Point(y, x), stroke, color, -1);
        }
        return image;
    }
}
