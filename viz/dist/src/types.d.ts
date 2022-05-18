export declare type Vector = Float32Array | number[];
export declare type Point2D = [number, number];
export declare type Point3D = [number, number, number];
export declare const enum InteractionMode {
    PAN = "PAN",
    SELECT = "SELECT"
}
export declare const enum RenderMode {
    POINT = "POINT",
    TEXT = "TEXT",
    SPRITE = "SPRITE"
}
export declare type Optional<T> = {
    [P in keyof T]?: T[P];
};
