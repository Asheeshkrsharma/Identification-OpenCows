import * as THREE from 'three';
export declare class LabelRenderParams {
    pointIndices: Float32Array;
    labelStrings: string[];
    scaleFactors: Float32Array;
    useSceneOpacityFlags: Int8Array;
    defaultFontSize: number;
    fillColors: Uint8Array;
    strokeColors: Uint8Array;
    constructor(pointIndices: Float32Array, labelStrings: string[], scaleFactors: Float32Array, useSceneOpacityFlags: Int8Array, defaultFontSize: number, fillColors: Uint8Array, strokeColors: Uint8Array);
}
export declare enum CameraType {
    Perspective = 0,
    Orthographic = 1
}
export declare class RenderContext {
    camera: THREE.Camera;
    cameraType: CameraType;
    cameraTarget: THREE.Vector3;
    screenWidth: number;
    screenHeight: number;
    nearestCameraSpacePointZ: number;
    farthestCameraSpacePointZ: number;
    backgroundColor: string;
    pointColors: Float32Array;
    pointScaleFactors: Float32Array;
    labels: LabelRenderParams | undefined;
    polylineColors: {
        [polylineIndex: number]: Float32Array;
    };
    polylineOpacities: Float32Array;
    polylineWidths: Float32Array;
    constructor(camera: THREE.Camera, cameraType: CameraType, cameraTarget: THREE.Vector3, screenWidth: number, screenHeight: number, nearestCameraSpacePointZ: number, farthestCameraSpacePointZ: number, backgroundColor: string, pointColors: Float32Array, pointScaleFactors: Float32Array, labels: LabelRenderParams | undefined, polylineColors: {
        [polylineIndex: number]: Float32Array;
    }, polylineOpacities: Float32Array, polylineWidths: Float32Array);
}
