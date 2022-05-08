import * as THREE from 'three';
import { ScatterPlotVisualizer } from './scatter_plot_visualizer';
import { RenderContext } from './render';
import { Sequence } from './data';
export declare class ScatterPlotVisualizerPolylines implements ScatterPlotVisualizer {
    id: string;
    private sequences;
    private scene;
    private polylines;
    private polylinePositionBuffer;
    private polylineColorBuffer;
    private pointSequenceIndices;
    getPointSequenceIndex(pointIndex: number): number | undefined;
    private updateSequenceIndices;
    private createPolylines;
    dispose(): void;
    setScene(scene: THREE.Scene): void;
    setSequences(sequences: Sequence[]): void;
    onPointPositionsChanged(newPositions: Float32Array): void;
    onRender(renderContext: RenderContext): void;
    onPickingRender(renderContext: RenderContext): void;
    onResize(newWidth: number, newHeight: number): void;
}
