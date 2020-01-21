#Jet Ryan Project

#2-Layer Neural Network

using Plots, Images, Distributions, LinearAlgebra
using Random
using HDF5
using Polynomials
using StatsPlots, SpecialFunctions

#trainingdata = h5read("ps3p4.h5", "data")

# linear activation function
hlin(a) = max(0,a)
# and it's derivative
hplin(a) = 1

# sigmoid activation function
hsig(a) = 1 ./ (1 .+ exp.(-a))
# and it's derivative
hpsig(a) = hsig(a) .* (1 .-hsig.(a))

# softmax activation
softmax(x) = exp.(x) ./ sum(exp.(x) .+ (10^-12))
# and its derivative
softpmax(x) = softmax(x) .* (Diagonal(ones(10,10)) .- softmax.(x)')

function change_one_hot(value)

    #value will be an number 0-9
    tempvec = zeros(10)
    tempvec[value + 1] = 1
    return tempvec

end

# activation function for output layer
hout = softmax#hlin
hpout = softpmax#hplin

# activation function for hidden layer
hhid = hsig
hphid = hpsig

function test(weight1, weight2, input, target, idx)

    N = length(idx)
    D = length(input[1])
    M = size(weight1)[1]

    error = 0.0
    # error_value = 0
    for n = 1:N

        x = input[idx[n]]
        t = target[idx[n]]
        #println(size(t))
        # forward propagate
        y = zeros(M+1)

        y[1] = 1.0 # bias node

        a = weight1 * x
        y[2:end] = hhid(a)
        a = 0
        a = weight2 * y
        z = hout(a)

        vector_holder = sum(((z .- t).^2))

        error += vector_holder

    end

    return error

end


function train(input, target)

    ep = 10^(-8)
    beta1 = 0.9
    beta2 =.999
    learning_rate = .0011

    # number of samples
    N = length(target)

    # dimension of input
    D = length(input[1])
    #println(D)

    O = 10 # number of classes in output

    # number to hold out
    Nhold = round(Int64, N/6)

    # number in training set
    Ntrain = N - Nhold

    # create indices
    idx = shuffle(1:N)

    trainidx = idx[1:Ntrain]
    testidx = idx[(Ntrain+1):N]

    println("$(length(trainidx)) training samples")
    println("$(length(testidx)) validation samples")

    # number of hidden nodes
    M = 300#800

    # batch size
    B = 256

    # input layer activation
    inputnode = zeros(D)

    # hidden layer activation
    hiddennode = zeros(M)

    # output node activation
    outputnode = zeros(O)

    # layer 1 weights
    weight1 = .01*randn(M, D)
    bestweight1 = weight1

    # layer 2 weights (inc bias)
    weight2 = .01*randn(O,M+1)
    bestweight2 = weight2

    numweights = prod(size(weight1)) + prod(size(weight2))
    println("$(numweights) weights")

    error = test(weight1, weight2, input, target, trainidx)
    println("Initial Training Error = $(error)")

    error = test(weight1, weight2, input, target, testidx)
    println("Initial Validation Error = $(error)")

    pdf = Uniform(1,Ntrain)

    error = []

    #stop = false
    cur_error = 1

    index = 1
    m1 = zeros(M, D)
    v1 = zeros(M, D)
    m2 = zeros(O, M+1)
    v2 = zeros(O, M+1)
    runtil = 0;
    while runtil < 600 || cur_error < .07

        grad1 = zeros(M, D)
        grad2 = zeros(O,M+1)

        for n = 1:B

            sample = trainidx[round(Int64, rand(pdf, 1)[1])]
            x = input[sample]
            t = target[sample]

            # forward propagate
            inputnode = x
            y = zeros(M+1)

            y[1] = 1 # bias node

            hiddennode = weight1 * inputnode
            y[2:end] = hhid(hiddennode)

            #outputnode = zeros(10)

            outputnode = weight2 * y
            z = hout(outputnode)
            # end forward propagate

            # output error
            delta = z.-t

            # compute layer 2 gradients by backpropagation of delta
            grad2[:,1] = delta*y[1]
            grad2[:,2:end] = hpout(z)* delta*(y[2:end])'
            #grad1 = (x*hphid(hiddennode)' .*(delta'*weight2[:,2:end]))'
            for i = 1:D
                for j = 1:M
                    grad1[j,i] += delta'*weight2[:,j+1]*hphid(hiddennode[j])*x[i]'
                end
            end

        end

        grad2 = grad2 / B
        grad1 = grad1 / B




        #adam backpropagation

        # update layer 2 weights
        m2 = beta1 .* m2 + (1 - beta1) .* grad2
        mt2 = m2 ./ (1 - (beta1 ^ index))
        v2 = beta2 .* v2 + (1 - beta2) .* (grad2 .^ 2)
        vt2 = v2 ./ (1 - (beta2 ^ index))
        weight2 += -learning_rate .* mt2 ./ (sqrt.(vt2) .+ ep)

        # update layer 1 weights
        m1 = beta1 .* m1 .+ (1 - beta1) .* grad1
        mt1 = m1 ./ (1 - (beta1 ^ index))
        v1 = beta2 .* v1 .+ (1 - beta2) .* (grad1 .^ 2)
        vt1 = v1 ./ (1 - (beta2 ^ index))
        weight1 += -learning_rate .* mt1 ./ (sqrt.(vt1) .+ ep)

        temp = test(weight1, weight2, input, target, testidx)
        push!(error, temp)

        cur_error = error_rate(weight1, weight2, input, target)
        cur_error = cur_error / N

        println("Batch Training Error = $(temp)")
        println("Current error rate = $(cur_error)")

        # window = 15
        # if length(error) > 2*window+1
        #     runningerror = error[(end-window):end]
        #
        #     mean1 = mean(runningerror)
        #     mean2 = mean(error[(end-2*window):(end-window)])
        #
        #     stop = (mean1) > mean2
        # end
        index = index + 1
        runtil = runtil + 1
    end

    error = test(weight1, weight2, input, target, trainidx)
    println("Final Training Error = $(error)")

    error = test(weight1, weight2, input, target, testidx)
    println("Final Validation Error = $(error)")

    return weight1, weight2, error
end

function error_rate(weight1, weight2, input, target)

    N = length(input)
    D = length(input[1])
    M = size(weight1)[1]

    #error = 0.0
    error_value = 0
    for n = 1:N
        x = input[n]
        t = target[n]

        # forward propagate
        y = zeros(M+1)

        y[1] = 1.0 # bias node

        a = weight1 * x
        y[2:end] = hhid.(a)

        a = weight2 * y
        z = hout(a)

        val = argmax(z)
        class = argmax(t)

        if(val != class)
            error_value = error_value + 1
        end

    end

    return error_value
end




function demo()

    #input, target = draw(300)
    h5open("mnist.h5", "r") do file
        labels = read(file, "train/labels")
        images = read(file, "train/images")

        input = []
        target = []

        N_size = size(labels)[1]

        for i = 1:N_size
            datanew = reshape(images[:,:,i], 784)
            #prepend!(datanew, 1)

                targetnew = change_one_hot(labels[i])
                #preprocess the data with a mean shift of 785
            push!(target, targetnew)
            push!(input, datanew)
        end
        w1, w2, err = train(input, target)

        #test_and_train_error_noh5(w1, w2)

        h5open("twolayer_10class.h5", "w") do file
            write(file, "weight1", w1)  # alternatively, say "@write file A"
            write(file, "weight2", w2)
            # write(file, "error", err)
        end
    end

end


function test_and_train_error()

    h5open("twolayer_10class.h5", "r") do file
        w1 = read(file, "weight1")  # alternatively, say "@write file A"
        w2 = read(file, "weight2")
        # err = read(file, "error")

        h5open("mnist.h5", "r") do file
            labels_train = read(file, "train/labels")
            images_train = read(file, "train/images")
            labels_test = read(file, "test/labels")
            images_test = read(file, "test/images")

            input_train = []
            target_train = []
            input_test = []
            target_test = []

            N_sizetrain = size(labels_train)[1]
            n_train = 0
            for i = 1:N_sizetrain

                #print
                n_train = n_train + 1
                datanew = reshape(images_train[:,:,i], 784)
                #prepend!(datanew, 1)
                targetnew = change_one_hot(labels_train[i])
                push!(target_train, targetnew)
                push!(input_train, datanew)
            end
            error_train = error_rate(w1, w2, input_train, target_train)

            N_sizetest = size(labels_test)[1]
            n_test = 0
            for i = 1:N_sizetest

                #print
                n_test = n_test + 1
                println(size(images_test))
                datanew = reshape(images_test[:,:,i], 784)
                #print(size(datanew))
                #prepend!(datanew, 1)
                targetnew = change_one_hot(labels_test[i])
                push!(target_test, targetnew)
                push!(input_test, datanew)
            end
            error_test = error_rate(w1, w2, input_test, target_test)

            print("The number of misclassifications for train is: ")
            println(error_train)
            print("The result for the error rate for train is: ")
            println(error_train/n_train)

            print("The number of misclassifications for test is: ")
            println(error_test)
            print("The result for the error rate for test is: ")
            println(error_test/n_test)

        end


    end

end

@time demo()
test_and_train_error()
