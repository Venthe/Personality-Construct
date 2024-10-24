# def generate_synthetic_data(input, input_parser=None):
#     """
#     Synthetic data refers to data generated via models or simulated environments e.g with LLM.
#     """
#     if input_parser is None:
#         raise Exception("Input parser has to be passed")

#     # training_nodes -> Text chunks trat represents coherent segments of the input data
#     #  nodes should also capture metadata i.e. neigbouring nodes content, references to previous and succeeding data
#     training_nodes = input_parser.parse_input(input)

# fmt: off
# Add tensorboard?
# Scratch learning would differ in:
#  - training dataset is predefined
#  - validation dataset (different from the training) is predefined
# TODO:
#  LoRA/QLoRA
#  AWQ/GPTQ 4 bit quantization
def llm_pipeline():
     # Number of passes through the entire dataset
     #  Too high risks overtraining
    epochs = 10
    learning_rate = 1e-5
    input_data = load_input_data() # E.g. word document
    validation_dataset = None
    validation_threshold = validation_dataset.treshold

    # expand the vocabulary when
    #  working with domain-specific names - medical, legal, scientific
    #  custom tokens - marker, segmentation
    #  out-of-vocabulary - ? is is a superset of domain specific names?
    provided_custom_tokens = set()

    # Step 1: Parse input data
    # Split input data into nodes with metadata
    #  that perserve the context e.g. preceeding and succeeding sentences in a document.
    training_nodes = parse_input_data(input_data)

    # Step 2: Generate synthetic dataset
    # Generate synthetic question-answer pairs for training later
    training_dataset = generate_synthetic_question_answer_embedding_pairs(training_nodes)

    # Step 3: Load model and tokenizer
    # Load any model to fine-tune e.g. LLM or embedding model.
    # ? context window
    # ? max new tokens
    # ? query wrapper prompt
    # ? tokenizer name
    # ? temperature - Model hyperparameter, are used to control model generation behavior
    # ? top_k - Model hyperparameter, are used to control model generation behavior
    # ? top_p - Model hyperparameter, are used to control model generation behavior
    def distributed_model(model):
        layers = model.layers

        def forward(input_tokens):
            return reduce(lambda output, layer: layer.forward(output), layers, input_tokens)
        
        def backward(loss):
            return sum([layer.backward(loss).get_gradients() for layer in reversed(layers)])/len(layers)
        
        def update(combined_gradients):
            [layer.update_weights(combined_gradients / len(layers)) for layer in layers]
            [layer.reset_gradients() for layer in layers]

    inference_model = load_model(model_name) # e.g., AutoModelForCausalLM.from_pretrained(model_name); or initialize a new model; or EmbeddingModel(load_model(model_name).get_input_embeddings())
    # Load tokenizer
    tokenizer_model = load_tokenizer(inference_model) # e.g., AutoTokenizer.from_pretrained(tokenizer_name)

    # Step 3.1: Expand the vocabulary of the tokenizer
    tokenizer_model.add_tokens(list(provided_custom_tokens))
    inference_model.resize_token_embedding_layer(len(tokenizer_model.vocabulary))
    for question, _ in training_dataset:
        question_tokens = tokenizer_model.tokenize(question)
        if tokenizer_model.unknown_token not in question_tokens:
            continue
        question_token_ids = tokenizer_model.convert_tokens_to_ids(question_tokens)
        new_tokens = set()

        encoded_tokens = tokenizer_model.encode(question, return_offsets_mapping=True, add_special_tokens=False)
        offsets = encoded_tokens['offset_mapping']
        for token, token_id, offset in zip(question_tokens, question_token_ids, offsets):
            if token == tokenizer_model.unknown_token:
                start, end = offset
                unknown_word = question[start:end]
                new_tokens.add(unknown_word)
                print(f"[UNK:{token_id}:{unknown_word}]")
            else:
                print(f"[{token_id}:{token}]")
        
        tokenizer_model.add_tokens(list(new_tokens))
        # Adjust the model embedding layer
        #  While the embedding layer is usually a single matrix, some models may have additional
        #  embeddings (e.g., position embeddings, type embeddings in BERT). However, token embeddings
        #  specifically are usually handled by a single layer in most architectures like GPT, BERT, etc.
        inference_model.resize_token_embedding_layer(len(tokenizer_model.vocabulary))

    # Step 4: Set up optimizer and loss function
    optimizer_scheduler = setup_optimizer_scheduler(load_optimizer_model(optimizer_name, learning_rate)) # torch.optim.AdamW
    loss_model = load_loss_function(loss_function_type) # torch.nn.CrossEntropyLoss

    # Step 5: Train the model (Finetune)
    # ? epochs
    # ? optimizers, torch.optim.AdamW ?
    # ? learning rate
    # ? loss function
    # ? bias
    training_info = {
        'epochs_trained': 0,
        'best_loss': float('inf')  # Start with infinity to track the best loss
    }
    epoch_log = []
    optimizer = None
    for epoch in range(epochs):
        optimizer = optimizer_scheduler.next()

        def process_epoch_with_data_parallelization(dataset, tokenizer_model, optimizer, inference_model, strategy):
            dataset_batches = to_batches(dataset)
            
            def process_batch(dataset_batch, strategy):
                total_gradients = None
                total_epoch_loss = 0
                for question, answer in dataset_batch:
                    if strategy[0] == "training":
                        strategy.optimizer.reset_gradients()
                    # Tokens are tensors - Tokens are PyTorch tensors
                    #  truncation, padding - handle input sequence length differences.
                    question_tokens = tokenizer_model.tokenize(question)
                    expected_answer_tokens = tokenizer_model.tokenize(answer)

                    # Forward pass (prediction)
                    output = inference_model.infer(question_tokens)

                    # Calculate loss
                    loss = loss_model.compute(output, expected_answer_tokens)

                    if strategy[0] == "training":
                        # Backward pass (propagate loss gradients)
                        total_gradients += compute_gradients(inference_model, loss)

                    # Update the state tracking
                    total_epoch_loss += tensor_to_float(loss)
                epoch_loss = total_epoch_loss / len(dataset_batch)
                average_gradients = total_gradients / len(dataset_batch)
                return epoch_loss, average_gradients

            gradients = parallelize(dataset_batches, process_batch, strategy)
            averaged_gradients = gradients / len(dataset_batches)

            # Update model weights using optimizer
            inference_model.update_weights(averaged_gradients, strategy.optimizer)

        training_epoch_loss = process_epoch_with_data_parallelization(dataset=training_dataset, inference_model=inference_model, tokenizer_model=tokenizer_model,strategy=("training", optimizer))
        training_info['epochs_trained'] += 1
        epoch_log.append((training_info, training_epoch_loss))

        if training_epoch_loss < training_info['best_loss']:
            training_info['best_loss'] = training_epoch_loss
        
        inference_model.save_checkpoint(epoch)
        tokenizer_model.save_checkpoint(epoch)
        optimizer.save_checkpoint(epoch)

        validation_epoch_loss = process_epoch_with_data_parallelization(dataset=validation_dataset, inference_model=inference_model, tokenizer_model=tokenizer_model,strategy=("validation"))
        if validation_epoch_loss < validation_threshold:
            raise Exception("We are departing from the goal")

        if training_epoch_loss < finetune_threshold:
            # Model is fine-tuned sufficiently
            break

    if training_info['epochs_trained'] > 0 and training_info['loss'] < threshold:
        model_checkpoint = inference_model.save_checkpoint('finetuned_model')
        tokenizer_checkpoint = tokenizer_model.save_checkpoint('finetuned_tokenizer')
        optimizer.save_checkpoint('finetuned_tokenizer')

        to_safetensors(model_checkpoint)
        to_safetensors(tokenizer_checkpoint)
