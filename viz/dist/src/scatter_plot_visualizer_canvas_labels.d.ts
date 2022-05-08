import * as THREE from 'three';
import { ScatterPlotVisualizer } from './scatter_plot_visualizer';
import { RenderContext } from './render';
import { Styles } from './styles';
export declare class ScatterPlotVisualizerCanvasLabels implements ScatterPlotVisualizer {
    private styles;
    id: string;
    private worldSpacePointPositions;
    private gc;
    private canvas;
    private labelsActive;
    constructor(container: HTMLElement, styles: Styles);
    private removeAllLabels;
    private makeLabels;
    private styleStringFromPackedRgba;
    onResize(newWidth: number, newHeight: number): void;
    dispose(): void;
    onPointPositionsChanged(newPositions: Float32Array): void;
    onRender(rc: RenderContext): void;
    setScene(scene: THREE.Scene): void;
    onPickingRender(renderContext: RenderContext): void;
}
