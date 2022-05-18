export interface Color {
    r: number;
    g: number;
    b: number;
    opacity: number;
}
export declare function parseColor(inputColorString: string): Color;
