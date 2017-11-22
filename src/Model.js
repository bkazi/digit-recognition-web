// @flow
import {
    Array1D,
    Array2D,
    CostReduction,
    Graph,
    AdamOptimizer,
    InGPUMemoryShuffledInputProviderBuilder,
    NDArrayMathGPU,
    Scalar,
    Session,
} from "deeplearn";
import mnist from "mnist";
import React from "react";

class Model extends React.Component {
    math = new NDArrayMathGPU();
    graph = new Graph();
    session = new Session(this.graph, this.math);
    inputTensor;
    labelTensor;
    costTensor;
    outputTensor;
    state = {
        ready: false,
    };

    constructor(props) {
        super(props);
        const layerShapes = this.props.layerShapes;
        this.inputTensor = this.graph.placeholder('input', [layerShapes[0]]);    
        this.labelTensor = this.graph.placeholder('output', [layerShapes[layerShapes.length - 1]]);
        
        const weights = new Array(layerShapes.length - 1);
        const biases = new Array(layerShapes.length - 1);
        const activations = new Array(layerShapes.length);
        activations[0] = this.inputTensor;
        
        for (let i = 1; i < layerShapes.length; i++) {
            weights[i] = this.graph.variable(`weights${i}`, Array2D.randNormal([layerShapes[i], layerShapes[i - 1]], 0, 0.1));
            biases[i] = this.graph.variable(`bias${i}`, Scalar.randNormal([layerShapes[i]], 0, 0.1));
            activations[i] = this.graph.relu(this.graph.add(this.graph.matmul(weights[i], activations[i - 1]), biases[i]));
        }
        
        this.outputTensor = activations[activations.length - 1];
        this.costTensor = this.graph.softmaxCrossEntropyCost(activations[activations.length - 1], this.labelTensor);
    }

    train(inputs, labels, optimizer, batchSize = inputs.length, iterations = 100) {
        const numOfBatches = inputs.length / batchSize;
        
        const shuffledInputProvider = new InGPUMemoryShuffledInputProviderBuilder([inputs, labels]);
        const [inputProvider, labelsProvider] = shuffledInputProvider.getInputProviders();
    
        const feedEntries = [
            {tensor: this.inputTensor, data: inputProvider},
            {tensor: this.labelTensor, data: labelsProvider},
        ];
    
        let avgCost = 0;
        const trainFunc = () => {
            const cost = this.session.train(this.costTensor, feedEntries, batchSize, optimizer, CostReduction.MEAN);
        
            avgCost += cost.get() / batchSize;
        }

        for (let j = 0; j < iterations; j++) {
            avgCost = 0;
            for (let i = 0; i < numOfBatches; i++) {
                this.math.scope(trainFunc);
            }
            console.log(`Iteration: ${j}, Cost: ${avgCost}`);
        }
    }

    predict = (input) => {
        let output;
        this.math.scope((keep, track) => {
            const testFeedEntries = [
                {tensor: this.inputTensor, data: input}
            ];

            output = keep(this.session.eval(this.outputTensor, testFeedEntries));
        });
        return output;
    }

    componentDidMount() {
        setTimeout(() => {
            const set = mnist.set(3000, 100);
            const training = set.training;
            let inputs = training.map(d => Array1D.new(d.input.map(i => i > 0.5 ? 1 : 0)));
            let labels = training.map(d => Array1D.new(d.output));
            this.train(inputs, labels, new AdamOptimizer(0.01, 0.9, 0.99), 32, 30);
            
            const testing = set.test;
            inputs = testing.map(d => Array1D.new(d.input.map(i => i > 0.5 ? 1 : 0)));
            labels = testing.map(d => Array1D.new(d.output));
            let hits = 0;
            for (let i = 0; i < inputs.length; i++) {
                const testOutput = this.predict(inputs[i]);
        
                if (this.math.argMaxEquals(testOutput, labels[i]).get()) {
                    hits++;
                }
            }
            console.log(`Accuracy: ${hits / inputs.length * 100}%`);
            this.setState((prevState) => ({
                ...prevState,
                ready: true,
            }));
        }, 0);
    }

    render() {
        return this.state.ready ? this.props.children(this.predict) : <div>Model is training</div>;
    }
}

export default Model;
