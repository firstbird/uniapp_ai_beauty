// import * as faceDetector from './models/detector';
// import * as faceParser from './models/parser';
import { FaceAlignmentDetector } from './detector/face_detector';
import { FaceParser } from './parser/face_parser';
import cv2 from '@/uni_modules/zj-opencv';

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

async function showFaceParser(imgPath: string, saveImg: boolean = true): Promise<[any, any, any]> {
    const im = await cv2.imread(imgPath);
    console.log(im.size());
    const [h, w] = [im.size()[0], im.size()[1]];
    const fp = new FaceParser();
    const parsingMap = fp.parseFace(im, null, false);
    const map = parsingMap[0].reshape(h, w, 1);

    const mask1 = cv2.compare(map, 10, cv2.CmpTypes.CMP_EQ);
    const mask2 = cv2.compare(map, 1, cv2.CmpTypes.CMP_EQ);
    const mask3 = cv2.compare(map, 14, cv2.CmpTypes.CMP_EQ);
    const mask = cv2.bitwiseOr(cv2.bitwiseOr(mask1, mask2), mask3).toUInt8();

    const imgMaskFg = cv2.bitwiseAnd(im, im, mask);
    const maskInv = cv2.bitwiseNot(mask.mul(255)).toUInt8();
    const imgMaskBg = cv2.bitwiseAnd(im, im, maskInv);

    const numOfClass = 17;
    if (saveImg) {
        const mapColor = new any.zeros(map.sizes, cv2.CV_8UC3);
        for (let pi = 1; pi <= numOfClass; pi++) {
            const index = map.threshold(pi, 10, 255, cv2.ThresholdTypes.THRESH_BINARY);
            mapColor.setTo(partColors[pi], index);
        }
        cv2.imwrite("./result/test_seg.jpg", mapColor);
        cv2.imwrite("./result/test_mask.jpg", mask.mul(255));
        cv2.imwrite("./result/img_mask_fg.jpg", imgMaskFg);
        cv2.imwrite("./result/img_mask_bg.jpg", imgMaskBg);
        console.log("Mask saved!");
    }
    return [imgMaskFg, imgMaskBg, mask];
}

// 以下函数未完全转换，仅提供了框架，需要根据 TypeScript 的规则进行进一步调整和修改
function fastGuideFilter(I: any, p: any, winSize: [number, number], eps: number, s: number): any {
    const h = I.rows;
    const w = I.cols;

    const size = [Math.round(w * s), Math.round(h * s)];
    const smallI = I.resize(new cv2.Size(size[0], size[1]), 0, 0, cv2.InterpolationFlags.INTER_CUBIC);
    const smallP = p.resize(new cv2.Size(size[0], size[1]), 0, 0, cv2.InterpolationFlags.INTER_CUBIC);

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

    const meanA = meanSmallA.resize(new cv2.Size(w, h), 0, 0, cv2.InterpolationFlags.INTER_LINEAR);
    const meanB = meanSmallB.resize(new cv2.Size(w, h), 0, 0, cv2.InterpolationFlags.INTER_LINEAR);

    let q = meanA.mul(I).add(meanB);

    return q;
}

function guideFilter(img: any): [any, any, any] {
    const guide = img;
    // const guide = img.cvtColor(cv2.COLOR_RGB2GRAY);
    const dst1 = cv2.guidedFilter(guide, img, 32, 2000, -1);
    const dst2 = cv2.guidedFilter(guide, img, 64, 1000, -1);
    const dst3 = cv2.guidedFilter(guide, img, 32, 1000, -1);

    return [dst1, dst2, dst3];
}

// 还有其他部分需要根据 TypeScript 的语法进行调整和修改
const imgPath: string = "./1.jpeg";
const [fg, bg, maskFg] = showFaceParser(imgPath, true);

const gray: any = fg.cvtColor(cv2.ColorConversionCodes.COLOR_BGR2GRAY);
const [mean, stddev] = cv2.meanStdDev(gray, maskFg);
console.log(mean, stddev);
const eps: number = stddev.at(0, 0) < 40 ? 0.001 : 0.01;
console.log(eps);
const winSize: [number, number] = [16, 16]; // convolution kernel

const I: any = fg.div(255); // Normalizing image
const p: any = I;
const s: number = 3; // step length

let guideFilterImg: any = fastGuideFilter(I, p, winSize, eps, s);
guideFilterImg = guideFilterImg.mul(255); // (0,1)->(0,255)
guideFilterImg = guideFilterImg.threshold(255, cv2.ThresholdTypes.THRESH_TRUNC).toUInt8();

const imgZero: any = new any.zerosLike(fg);
const { contours, hierarchy } = cv2.findContours(gray, cv2.RetrModes.RETR_TREE, cv2.ContourApproximationModes.CHAIN_APPROX_SIMPLE);
cv2.drawContours(imgZero, contours, -1, new cv2.Vec(255, 255, 255), 3);

const blurredImg: any = guideFilterImg;

const output: any = imgZero.threshold(5, 255, cv2.ThresholdTypes.THRESH_BINARY).toUInt8().eq(new cv2.Vec(255, 255, 255))
    .bitwiseAnd(cv2.GaussianBlur(blurredImg, new cv2.Size(5, 5), 0))
    .add(cv2.bitwiseAnd(blurredImg, cv2.bitwiseNot(imgZero.threshold(5, 255, cv2.ThresholdTypes.THRESH_BINARY).toUInt8())))
    .toUInt8();

cv2.imwrite("./result/mask.jpg", imgZero);
cv2.imwrite("./result/post.jpg", output);
cv2.imwrite("./result/winSize_16.jpg", guideFilterImg);
