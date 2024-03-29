import * as sim from "lib-simulation_wasm";

const simulation = new sim.Simulation();

var gene = 0;
document.getElementById('train').onclick = function() {
    gene += 1;
    console.log(simulation.train() + ", gene=" + gene);
};

const viewport = document.getElementById('viewport');

const viewportWidth = viewport.width;
const viewportHeight = viewport.height;

const viewportScale = window.devicePixelRatio || 1;
viewport.width = viewportWidth * viewportScale;
viewport.height = viewportHeight * viewportScale;

viewport.style.width = viewportWidth + 'px';
viewport.style.height = viewportHeight + 'px';

const ctxt = viewport.getContext('2d');

ctxt.scale(viewportScale, viewportScale);

CanvasRenderingContext2D.prototype.drawTriangle =
    function (x, y, size, rotation) {
        this.beginPath();

        this.moveTo(
            x + Math.cos(rotation) * size * 1.5,
            y + Math.sin(rotation) * size * 1.5,
        );

        this.lineTo(
            x + Math.cos(rotation + 2.0 / 3.0 * Math.PI) * size,
            y + Math.sin(rotation + 2.0 / 3.0 * Math.PI) * size,
        );

        this.lineTo(
            x + Math.cos(rotation + 4.0 / 3.0 * Math.PI) * size,
            y + Math.sin(rotation + 4.0 / 3.0 * Math.PI) * size,
        );

        this.lineTo(
            x + Math.cos(rotation) * size * 1.5,
            y + Math.sin(rotation) * size * 1.5,
        );

        this.stroke();

        this.fillStyle = 'rgb(255, 255, 255)'; // A nice white color
        this.fill();
    };

CanvasRenderingContext2D.prototype.drawCircle =
    function(x, y, radius) {
        this.beginPath();

        this.arc(x, y, radius, 0, 2.0 * Math.PI);

        this.fillStyle = 'rgb(0, 255, 128)';
        this.fill();
    };

function redraw() {
    ctxt.clearRect(0, 0, viewportWidth, viewportHeight);

    // Performs 10 steps per frame, which makes simulation 10x faster
    // (at least if your computer can catch up!)
    for (let i = 0; i < 10; i += 1) {
        simulation.step();
    }
    // simulation.step();

    const world = simulation.world();

    for (const food of world.foods) {
        ctxt.drawCircle(
            food.x * viewportWidth,
            food.y * viewportHeight,
            (0.01 / 2.0) * viewportWidth,
        );
    }

    for (const animal of simulation.world().animals) {
        ctxt.drawTriangle(
            animal.x * viewportWidth,
            animal.y * viewportHeight,
            0.01 * viewportWidth,
            animal.rotation,
        );
    }

    // requestAnimationFrame() schedules code only for the next frame.
    //
    // Because we want for our simulation to continue forever, we've
    // gotta keep re-scheduling our function:
    requestAnimationFrame(redraw);
}

redraw();

