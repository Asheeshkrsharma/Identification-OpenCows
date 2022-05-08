import * as THREE from 'three';
import { Point2D } from './types';
export declare function vector3DToScreenCoords(cam: THREE.Camera, w: number, h: number, v: THREE.Vector3): Point2D;
export declare function vector3FromPackedArray(a: Float32Array, pointIndex: number): THREE.Vector3;
export declare function getNearFarPoints(worldSpacePoints: Float32Array, cameraPos: THREE.Vector3, cameraTarget: THREE.Vector3): [number, number];
export declare function createTextureFromCanvas(image: HTMLCanvasElement): THREE.Texture;
export declare function createTextureFromImage(image: HTMLImageElement, onImageLoad: () => void): THREE.Texture;
export declare function hasWebGLSupport(): boolean;
export declare function extent(data: number[]): number[];
export declare function scaleLinear(value: number, domain: number[], range: number[]): number;
export declare function scaleExponential(value: number, domain: number[], range: number[]): number;
export declare function packRgbIntoUint8Array(rgbArray: Uint8Array, labelIndex: number, r: number, g: number, b: number): void;
export declare function styleRgbFromHexColor(hex: number | string): [number, number, number];
export declare function getDefaultPointInPolylineColor(index: number, totalPoints: number, startHue: number, endHue: number, saturation: number, lightness: number): THREE.Color;