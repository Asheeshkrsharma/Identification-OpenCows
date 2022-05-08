import { Points } from '../../src/data';
export interface Data {
    labels: number[];
    labelNames: string[];
    projection: Points;
    path?: string[];
}
export declare const data: Data;
