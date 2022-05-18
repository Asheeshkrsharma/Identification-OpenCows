export interface PointMetadata {
    label?: string;
    [key: string]: number | string | undefined;
}
export interface SpriteMetadata {
    spriteImage?: HTMLImageElement | string;
    singleSpriteSize: [number, number];
    spriteIndices?: number[];
}
export interface Sequence {
    indices: number[];
}
export declare type Point2D = [number, number];
export declare type Point3D = [number, number, number];
export declare type Points = Array<Point2D | Point3D>;
export declare class Dataset {
    points: Points;
    metadata: PointMetadata[];
    spriteMetadata?: SpriteMetadata;
    dimensions: number;
    constructor(points: Points, metadata?: PointMetadata[]);
    setSpriteMetadata(spriteMetadata: SpriteMetadata): void;
}
