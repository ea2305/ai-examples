class Pointer {

  constructor (x = 0, y = 0, size = 25) {
    this.x = x
    this.y = y
    this.size = size
  }

  showInFrame () {
    ellipse(
      map(this.x, 0, 1, 0, width), // x axis
      map(this.y, 1, 0, 1, height), // y axis
      this.size
    )
  }
}