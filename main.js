// variables
let pointers = [] // { x: '?', y: '?' } 

let m, b;

// Optimazer and learning rate
const learningRate = 0.01;
const optimizer = tf.train.sgd(learningRate);

function setup () {
  createCanvas(500, 400)

  // initialize m and b
  m = tf.variable(
    tf.scalar(random(1))
  )

  b = tf.variable(
    tf.scalar(random(1))
  )
}

/**
 * Insert Point into list
 */
function mousePressed() {
  pointers.push(
    new Pointer(
      map(mouseX, 0, width, 0, 1), // x axis
      map(mouseY, 0, height, 1, 0), // y axis
      15 // size
    )
  )
}

function draw () {
  background(0)

  if (pointers.length > 0) {

    tf.tidy(() => { // Clean memory
      // generate Y pointer tensor
      const ys = tf.tensor1d(
        pointers.map(point => point.y)
      )

      // update Network
      optimizer.minimize(() => loss(networkOut(
        pointers.map(point => point.x)
      ), ys))
    })

  }

  for (let i = 0; i < pointers.length; i++) {
    // show points
    pointers[i].showInFrame()
  }

  // generate line
  const lineXs = [0,1]
  const tensorYs = tf.tidy(() => networkOut(lineXs)) // Clean up memory

  let x1 = map(lineXs[0], 0, 1, 0, width) // begin point X line
  let x2 = map(lineXs[1], 0, 1, 0, width) // end point X line

  // convert tensor to Array
  let lineYs = tensorYs.dataSync()
  
  tensorYs.dispose() // Clean up memory

  let y1 = map(lineYs[0], 0, 1, height, 0) // begin point Y line
  let y2 = map(lineYs[1], 0, 1, height, 0) // end point Y line

  stroke('blue')
  strokeWeight(5)
  // draw line
  line(x1, y1, x2, y2)
  noStroke()
}

/**
 * Generate a output of network
 * @param {Array} vector : Pointer X/Y Axis
 */
function networkOut (vector) {
  const xs = tf.tensor1d(vector)
  
  const ys = xs.mul(m)
              .add(b)

  return ys
}

// y = a * x^2 + b * x + c.
function loss (pred, label) {
  return pred.sub(label).square().mean()
}