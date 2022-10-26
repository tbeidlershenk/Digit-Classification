const canvas = document.getElementById('canv');
const toolbar = document.getElementById('toolbar');
const ctx = canvas.getContext('2d');
const right = document.getElementById('test');

const canvasOffsetX = canvas.offsetLeft;
const canvasOffsetY = canvas.offsetTop;

canvas.width = window.innerWidth - canvasOffsetX;
canvas.height = window.innerHeight - canvasOffsetY;
imgData = ctx.createImageData(canvas.width, canvas.height);
console.log(canvas.width)
console.log(canvas.height)

let isPainting = false;
let lineWidth = 15;
let startX;
let startY;

function clear() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

toolbar.addEventListener('click', e => {
    if (e.target.id === 'clear') {
        console.log("Clearing canvas");
        clear();
    }
});

toolbar.addEventListener('change', e => {
    if(e.target.id === 'stroke') {
        ctx.strokeStyle = e.target.value;
    }

    if(e.target.id === 'lineWidth') {
        lineWidth = e.target.value;
    }
    
});

const draw = (e) => {
    if(!isPainting) {
        return;
    }
    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';
    ctx.lineTo(e.clientX - canvasOffsetX, e.clientY);
    console.log(e.clientX)
    console.log(e.clientY)

    ctx.stroke();
}

canvas.addEventListener('mousedown', (e) => {
    isPainting = true;
    startX = e.clientX;
    startY = e.clientY;
});

canvas.addEventListener('mouseup', (e) => {
    isPainting = false;
    ctx.stroke();
    ctx.beginPath();
});

canvas.addEventListener('mousemove', draw);

document.addEventListener('keydown', (e) => {
    if (e.key == 'c') {
        clear();
    }
})

document.addEventListener('keydown', (e) => {
    imgData = ctx.getImageData(0,0,canvas.width, canvas.height);
    if (e.key == 's') {
        console.log(imgData)
    }
});