import React from "react";
import {
    Array1D,
    NDArrayMathCPU,
} from "deeplearn";

const size = 112;

class Canvas extends React.Component {
    isDrawing = false;
    resizedCanvas = document.createElement("canvas");
    resizedContext = this.resizedCanvas.getContext("2d");

    constructor() {
        super();
        this.resizedCanvas.height = "28";
        this.resizedCanvas.width = "28";
    }

    componentDidMount() {
        this.ctx = this.canvasRef.getContext("2d");
        this.canvasRef.width = size;
        this.canvasRef.height = size;
    }

    draw = () => {
        const ctx = this.ctx;
        ctx.beginPath();
        ctx.moveTo(this.prevX, this.prevY);
        ctx.lineTo(this.currX, this.currY);
        ctx.strokeStyle = "red";
        ctx.lineWidth = 4;
        ctx.stroke();
        ctx.closePath();
    }

    mouseDown = (e) => {
        this.isDrawing = true;
        this.currX = e.clientX - this.canvasRef.offsetLeft;
        this.currY = e.clientY - this.canvasRef.offsetTop;

    }

    mouseMove = (e) => {
        if (this.isDrawing) {
            this.prevX = this.currX;
            this.prevY = this.currY;
            this.currX = e.clientX - this.canvasRef.offsetLeft;
            this.currY = e.clientY - this.canvasRef.offsetTop;
            this.draw();
        }
    }

    mouseUp = (e) => {
        this.isDrawing = false;
    }

    submit = () => {
        this.resizedContext.clearRect(0, 0, 28, 28);
        this.resizedContext.drawImage(this.canvasRef, 0, 0, 28, 28);
        const data = this.resizedContext.getImageData(0, 0, 28, 28).data;

        this.imgRef.src = this.resizedCanvas.toDataURL();

        const binData = [];
        for (let i = 0; i < data.length; i += 4) {
            binData[i/4] = data[i] > 0.5 ? 1 : 0;
        }
        const math = new NDArrayMathCPU();
        console.log(math.argMax(this.props.predict(Array1D.new(binData))).get());
    }

    clear = () => {
        this.ctx.clearRect(0, 0, size, size);
    }

    render() {
        return <div>
            <canvas
                ref={(ref) => {
                    this.canvasRef = ref;
                }}
                width={size}
                height={size}
                onMouseMove={this.mouseMove}
                onMouseDown={this.mouseDown}
                onMouseUp={this.mouseUp}
                style={{border: "1px solid black"}}
            />
            <button onClick={this.submit}>Predict</button>
            <button onClick={this.clear}>Clear</button>
            <img alt="" ref={(ref) => {this.imgRef = ref}} />
        </div>;
    }
}

export default Canvas;