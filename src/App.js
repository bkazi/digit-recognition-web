// @flow
import React, { Component } from 'react';
import './App.css';

import Model from "./Model";
import Canvas from "./Canvas";

class App extends Component {
  render() {
    return (
      <div className="App">
        <Model layerShapes={[784, 128, 128, 10]}>
          {(predict) => {
            return (
              <div>
                <span>Hi</span>
                <Canvas predict={predict}/>
              </div>
            );
          }}
        </Model>
      </div>
    );
  }
}

export default App;
