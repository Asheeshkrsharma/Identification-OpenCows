export declare type Color = string;
export interface LabelStyles {
    fontSize: number;
    scaleDefault: number;
    scaleLarge: number;
    fillColorSelected: Color;
    fillColorHover: Color;
    strokeColorSelected: Color;
    strokeColorHover: Color;
    strokeWidth: number;
    fillWidth: number;
}
export interface Label3DStyles {
    fontSize: number;
    scale: number;
    color: Color;
    backgroundColor: Color;
    colorUnselected: Color;
    colorNoSelection: Color;
}
export interface PointStyles {
    colorUnselected: Color;
    colorNoSelection: Color;
    colorSelected: Color;
    colorHover: Color;
    scaleDefault: number;
    scaleSelected: number;
    scaleHover: number;
}
export interface FogStyles {
    color: Color;
    enabled: boolean;
    threshold: number;
}
export interface PolylineStyles {
    startHue: number;
    endHue: number;
    saturation: number;
    lightness: number;
    defaultOpacity: number;
    defaultLineWidth: number;
    selectedOpacity: number;
    selectedLineWidth: number;
    deselectedOpacity: number;
}
export interface SelectStyles {
    fill: Color;
    fillOpacity: number;
    stroke: Color;
    strokeWidth: number;
    strokeDashArray: string;
}
export interface SpritesStyles {
    minPointSize: number;
    imageSize: number;
    colorUnselected: Color;
    colorNoSelection: Color;
}
export interface Styles {
    backgroundColor: Color;
    axesVisible: boolean;
    fog: FogStyles;
    label: LabelStyles;
    label3D: Label3DStyles;
    point: PointStyles;
    polyline: PolylineStyles;
    select: SelectStyles;
    sprites: SpritesStyles;
}
export interface UserStyles {
    backgroundColor?: number;
    axesVisible?: boolean;
    fog?: Partial<FogStyles>;
    label?: Partial<LabelStyles>;
    label3D?: Partial<Label3DStyles>;
    point?: Partial<PointStyles>;
    polyline?: Partial<PolylineStyles>;
    select?: Partial<SelectStyles>;
    sprites?: Partial<SpritesStyles>;
}
export declare function makeStyles(userStyles: UserStyles | undefined): Styles;
