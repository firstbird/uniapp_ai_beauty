// import * as faceDetector from './models/detector';
// import * as faceParser from './models/parser';
import { FaceAlignmentDetector } from './detector/face_detector';
import { FaceParser } from './parser/face_parser';
import cv2 from '@/uni_modules/zj-opencv';
import { Mat, MatVector} from '@/uni_modules/zj-opencv/js_sdk/opencv';


import * as fs from 'fs';

const partColors: [number, number, number][] = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0],
    [255, 0, 85], [255, 0, 170],
    [0, 255, 0], [85, 255, 0], [170, 255, 0],
    [0, 255, 85], [0, 255, 170],
    [0, 0, 255], [85, 0, 255], [170, 0, 255],
    [0, 85, 255], [0, 170, 255],
    [255, 255, 0], [255, 255, 85], [255, 255, 170],
    [255, 0, 255], [255, 85, 255], [255, 170, 255],
    [0, 255, 255], [85, 255, 255], [170, 255, 255]
];

async function showFaceBbox(imgPath: string): Promise<void> {
    if (!fs.existsSync("./result")) {
        console.log("make dir!");
        fs.mkdirSync("./result");
    }

    const im = await cv2.imread(imgPath);
    const fd = new FaceAlignmentDetector();
    const bboxes = fd.detectFace(im, false);
    const ret = bboxes[0].slice(0, 4);
    console.log(ret);

    //cv2.rectangle(im, new cv2.Rect(ret[1], ret[0], ret[3] - ret[1], ret[2] - ret[0]), new cv2.Vec(0, 255, 0), 2);
    cv2.rectangle(im, new cv2.Point(ret[1], ret[0]), new cv2.Point(ret[3] - ret[1], ret[2] - ret[0]), new cv2.Vec(0, 255, 0), 2);
	const score = bboxes[0][bboxes[0].length - 1];

	// todo mzl
    // cv2.write("./result/test_bbox.jpg", im);
}

async function showFaceParser(imgPath: string, saveImg: boolean = true): Promise<[Mat, Mat, Mat]> {
    const im = await cv2.imread(imgPath);
    console.log(im.size());
    const [h, w] = [im.rows, im.cols];
    const fp = new FaceParser();
    const parsingMap = fp.parseFace(im, null, false);
    const map = parsingMap[0].reshape(h, w, 1);

	const mask1 = new cv2.Mat();
	const mat1 = new cv2.Mat.ones(map.rows, map.cols, map.type);
	const eye = cv2.Mat.eye(map.rows, map.cols);

    cv2.compare(map, mat1.mul(eye, 10), mask1, cv2.CMP_EQ);
	
	const mask2 = new cv2.Mat();	
    cv2.compare(map, mat1, mask2, cv2.CMP_EQ);
	
	const mask3 = new cv2.Mat();
    cv2.compare(map, mat1.mul(eye, 14), mask1, cv2.CMP_EQ);

    const mask4 = new cv2.Mat();
	cv2.bitwise_or(mask1, mask2, mask4);
	const mask = new cv2.Mat();
    cv2.bitwise_or(mask4, mask3, mask);// as uint8 todo mzl

    let maskSum1 = new cv2.Mat();
    cv2.add(mask1, mask2, maskSum1);
    let maskSum2 = new cv2.Mat();
    cv2.add(maskSum1, mask3, maskSum2);

    maskSum2.convertTo(mask, cv2.CV_8UC1);

	const imgMaskFg = new cv2.Mat();
    cv2.bitwise_and(im, im, imgMaskFg, mask);
	const maskInv = new cv2.Mat();
    cv2.bitwise_not(mask.mul(eye, 255), maskInv);
	const imgMaskBg = new cv2.Mat();
	// mask = cv2.GaussianBlur(mask, (5, 5), 0)
	cv2.GaussianBlur(mask, mask, new cv2.Size(5, 5), 0, 0, cv2.BORDER_DEFAULT);
    cv2.bitwise_and(im, im, imgMaskBg, maskInv);

    const numOfClass = 17;
    if (saveImg) {
        const mapColor = new cv2.Mat.zeros(map.sizes, cv2.CV_8UC3);
        for (let pi = 1; pi <= numOfClass; pi++) {
            const index = map.threshold(pi, 10, 255, cv2.THRESH_BINARY);
            mapColor.setTo(new cv2.Scalar(partColors[pi][0], partColors[pi][1], index));
        }
		// todo mzl
        // cv2.imwrite("./result/test_seg.jpg", mapColor);
        // cv2.imwrite("./result/test_mask.jpg", mask.mul(255));
        // cv2.imwrite("./result/img_mask_fg.jpg", imgMaskFg);
        // cv2.imwrite("./result/img_mask_bg.jpg", imgMaskBg);
        console.log("Mask saved!");
    }
    return [imgMaskFg, imgMaskBg, mask];
}

// 以下函数未完全转换，仅提供了框架，需要根据 TypeScript 的规则进行进一步调整和修改
function fastGuideFilter(I: any, p: any, winSize: [number, number], eps: number, s: number): any {
    const h = I.rows;
    const w = I.cols;

    const size = [Math.round(w * s), Math.round(h * s)];
    const smallI = I.resize(new cv2.Size(size[0], size[1]), 0, 0, cv2.INTER_CUBIC);
    const smallP = p.resize(new cv2.Size(size[0], size[1]), 0, 0, cv2.INTER_CUBIC);

    const X = winSize[0];
    const smallWinSize = [Math.round(X * s), Math.round(X * s)];

    const meanSmallI = smallI.blur(smallWinSize);
    const meanSmallP = smallP.blur(smallWinSize);

    const meanSmallII = smallI.mul(smallI).blur(smallWinSize);
    const meanSmallIp = smallI.mul(smallP).blur(smallWinSize);

    const varSmallI = meanSmallII.sub(meanSmallI.mul(meanSmallI));
    const covSmallIp = meanSmallIp.sub(meanSmallI.mul(meanSmallP));

    let smallA = covSmallIp.div(varSmallI.add(eps));
    let smallB = meanSmallP.sub(smallA.mul(meanSmallI));

    const meanSmallA = smallA.blur(smallWinSize);
    const meanSmallB = smallB.blur(smallWinSize);

    const meanA = meanSmallA.resize(new cv2.Size(w, h), 0, 0, cv2.INTER_LINEAR);
    const meanB = meanSmallB.resize(new cv2.Size(w, h), 0, 0, cv2.INTER_LINEAR);

    let q = meanA.mul(I).add(meanB);

    return q;
}

function guideFilter(img: any): [any, any, any] {
    const guide = img;
    // const guide = img.cvtColor(cv2.COLOR_RGB2GRAY);
	const dst1 : Mat = new cv2.Mat();
    const dst2 : Mat = new cv2.Mat();
    const dst3 : Mat = new cv2.Mat();
	// todo mzl
    // cv2.guidedFilter(guide, img, 32, 2000, -1);
    // const dst2 = cv2.guidedFilter(guide, img, 64, 1000, -1);
    // const dst3 = cv2.guidedFilter(guide, img, 32, 1000, -1);

    return [dst1, dst2, dst3];
}

// 还有其他部分需要根据 TypeScript 的语法进行调整和修改
const imgPath: string = "./1.jpeg";
const [fg, bg, mask_fg] = await showFaceParser(imgPath, true);


let gray = new cv2.Mat();
cv2.cvtColor(fg, gray, cv2.COLOR_BGR2GRAY);
let variances = new cv2.Mat();
let stddev = new cv2.Mat();
cv2.meanStdDev(gray,  variances, stddev, mask_fg);
console.log(variances);
// todo mzl
const eps = stddev.intAt(1) < 40 ? 0.001 : 0.01;
console.log(eps);
const winSize: [number, number] = [16, 16]; // convolution kernel

const I = fg.mul(cv2.Mat.eye(fg.cols, fg.rows, cv2.CV_8UC1), 1 / 255.0);
const p: any = I;
const s: number = 3; // step length

let guideFilterImg: any = fastGuideFilter(I, p, winSize, eps, s);
guideFilterImg = guideFilterImg.mul(255); // (0,1)->(0,255)
guideFilterImg = guideFilterImg.threshold(255, cv2.THRESH_TRUNC).toUInt8();

// const imgZero: any = new any.zerosLike(fg);
const imgZero = new cv2.Mat.zeros(fg.size(), fg.type());
// const { contours, hierarchy } = cv2.findContours(gray, cv2.RetrModes.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE);
// cv2.drawContours(imgZero, contours, -1, new cv2.Vec(255, 255, 255), 3);
let binary = new cv2.Mat();
const threshold = 5;

cv2.threshold(gray, binary, threshold, 255, cv2.THRESH_BINARY);
let hierarchy = new cv2.Mat();
let contours = new cv2.MatVector();
cv2.findContours(binary, contours, hierarchy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE);
cv2.drawContours(imgZero, contours, -1, new cv2.Scalar(255, 255, 255), 3);

const blurredImg: any = guideFilterImg;

let output = new cv2.Mat();
if (imgZero.intAt(0) == 255 && imgZero.intAt(1) == 255 && imgZero.intAt(2) == 255) {
  cv2.GaussianBlur(blurredImg, output, new cv2.Size(5, 5), 0, 0, cv2.BORDER_DEFAULT);
} else {
  output = blurredImg;
}

let binaryImg = new cv2.Mat();
cv2.threshold(imgZero, binaryImg, 255, 255, cv2.THRESH_BINARY);

// cv2.imwrite("./result/mask.jpg", imgZero);
// cv2.imwrite("./result/post.jpg", output);
// cv2.imwrite("./result/winSize_16.jpg", guideFilterImg);
