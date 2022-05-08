import { Styles } from './styles';
export interface ScatterBoundingBox {
    x: number;
    y: number;
    width: number;
    height: number;
}
export declare class ScatterPlotRectangleSelector {
    private svgElement;
    private rectElement;
    private isMouseDown;
    private startCoordinates;
    private lastBoundingBox;
    private selectionCallback;
    constructor(container: HTMLElement, selectionCallback: (boundingBox: ScatterBoundingBox) => void, styles: Styles);
    onMouseDown(offsetX: number, offsetY: number): void;
    onMouseMove(offsetX: number, offsetY: number): void;
    onMouseUp(): void;
}
