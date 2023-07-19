import streamlit as st
import pandas as pd
from skllm import ZeroShotGPTClassifier, MultiLabelZeroShotGPTClassifier, FewShotGPTClassifier
from skllm.config import SKLLMConfig

SKLLMConfig.set_openai_key("YOUR API")
SKLLMConfig.set_openai_org("YOUR ORG")

# Initialize the classifiers
clf_zero_shot = ZeroShotGPTClassifier(openai_model="gpt-3.5-turbo")
clf_multi_label = MultiLabelZeroShotGPTClassifier(openai_model="gpt-3.5-turbo")
clf_few_shot = FewShotGPTClassifier(openai_model="gpt-3.5-turbo")

# Classification options
classification_options = {
    "About Scikit-LLM": None,
    "Zero Shot": clf_zero_shot,
    "Multi-label Zero Shot": clf_multi_label,
    "Few-Shot": clf_few_shot,
    
}



# Dropdown for selecting the classification option
classification_option = st.selectbox("Select the classification option:", list(classification_options.keys()))

if classification_option == "About Scikit-LLM":
    st.markdown("### Scikit-LLM")
    st.markdown("NLP is an important frontier in data science. However, it has its limitations. We’ve built some pretty cool libraries and techniques, but there’s a lot room for improvement. This is where large language models, or LLMs for short, come in. They give us a chance to push NLP even further.")
    st.markdown("Scikit-learn, a well-known player in the machine learning field, supercharged with the cutting-edge technology of large language models. Now, that’s exactly where scikit-llm comes in. It takes the power of OpenAI’s API and wraps it up in a familiar sklearn package.")
    st.markdown("### Know More About Scikit-LLM")
    st.markdown("github.com/iryna-kondr/scikit-llm")


if classification_option == "Zero Shot":
    # Check if the tutorial checkbox is checked and the tutorial has not been displayed yet

        # Display the tutorial text
    st.markdown("### Zero Shot Classifier Tutorial")
    st.markdown("The ZeroShotClassifier is a type of classifier. It allows you to input candidate labels of your choice where texts can be classified based on the provided labels.")
    st.markdown("For example, you can include positive and negative categories. If a statement says that the product is terrible, the classifier can understand the meaning of what 'negative' means and label the statement as negative.")
    st.markdown("#### How to Use:")
    st.markdown("1. Input your candidate labels as a comma-separated list.")
    st.markdown("2. Input your texts to be classified. Separate them with line breaks.")
    st.markdown("3. Press the button to classify the texts, and the results will be displayed in a data frame.")

    # Get candidate labels from user
    default_candidate_labels = "Positive, Negative, Neutral"
    candidate_labels_input = st.text_input("Enter candidate labels as a comma-separated list (e.g., positive, negative, neutral):", default_candidate_labels)
    candidate_labels = candidate_labels_input.split(',')

    # Get user input as a block of text
    default_user_input = "I really love this product!\nI hate this product so much!\nI have nothing to say."
    user_input = st.text_area("Enter the texts to be classified (one per line)", default_user_input)

    # Split the block of text into individual lines
    user_inputs = user_input.splitlines()

    # Create a button to run the classification
    if st.button("Run"):
        # Fit the classifier with some dummy data and the user-provided candidate labels
        clf_zero_shot.fit(["dummy"], candidate_labels)

        # Classify the user inputs
        results = []
        for user_input in user_inputs:
            sentiment = clf_zero_shot.predict([user_input])[0]
            results.append({'Text': user_input, 'Sentiment': sentiment})

        # Create a dataframe from the results
        results_df = pd.DataFrame(results)

        # Display the results
        st.dataframe(results_df)



if classification_option == "Multi-label Zero Shot":
    st.markdown("### Multi Label Zero Shot Classifier")
    st.markdown("MultiLabelZeroShotClassifier allows you to input multiple candidate labels of your choice. A text can have multiple labels.")
    st.markdown("For example, if you have a statement talking about how they recommend the product because of the quality and price, given that you have these labels, the statement will be classified under both 'quality' and 'price'.")
    st.markdown("#### How to Use:")
    st.markdown("1. Input your candidate labels as a comma-separated list.")
    st.markdown("2. Input your texts to be classified. Separate them with line breaks.")
    st.markdown("3. Press the button to classify the texts, and the results will be displayed in a data frame.")

    st.markdown("### Input Your Own Text")

    # Get candidate labels from user
    default_candidate_labels = "Quality, Customer Service, Price"
    candidate_labels_input = st.text_input("Enter candidate labels as a comma-separated list:", default_candidate_labels)
    candidate_labels = candidate_labels_input.split(',')

    # Get user input as a block of text
    default_user_input = "The materials of this product is extremely nice. I can't believe it's this cheap!\nI hate how it takes so long before the company responds to my complaints.\n"
    user_input = st.text_area("Enter the texts to be classified (one per line):", default_user_input)

    # Split the block of text into individual lines
    user_inputs = user_input.splitlines()

    # Create a button to run the classification
    if st.button("Run"):
        # Fit the classifier with the user-provided candidate labels
        clf_multi_label.fit(None, [candidate_labels])

        # Classify the user inputs
        results = []
        for user_input in user_inputs:
            labels = clf_multi_label.predict([user_input])[0]
            results.append({'Text': user_input, 'Labels': ', '.join(labels)})

        # Create a dataframe from the results
        results_df = pd.DataFrame(results)

        # Display the results
        st.dataframe(results_df)


elif classification_option == "Few-Shot":
    st.markdown("### Few Shot Classifier Tutorial")
    st.markdown("The Few Shot Classifier allows you to input candidate labels and sample texts for additional context.")
    st.markdown("By providing example texts classified as ‘satisfied’ or ‘dissatisfied,’ the classifier learns to understand the context of the texts and their corresponding labels. Once trained, it can classify new texts based on whether they are categorized as ‘satisfied’ or ‘dissatisfied’ according to the learned context.")
    st.markdown("#### How to Use:")
    st.markdown("1. Enter the number of candidate labels")
    st.markdown("2. Input the name of the label per class")
    st.markdown("3. Include what are the sample texts per label")
    st.markdown("4. Input the texts to be classified")
    st.markdown("5. Press the button to generate classified texts")
    st.markdown("#### Input your own text:")

    # Default values
    default_labels = ["Satisfied", "Dissatisfied"]
    default_texts = [
        ["I love this product so much. I will recommend it to my family",
         "Spending this much is so worth it because of the built quality",
         "I don't regret buying this product!"],
        ["It doesn't work well, this product is so bad.",
         "The quality is not good at all, it feels so flimsy.",
         "I wasted my money on this!"]
    ]

    default_input_to_classify = [
        "Hoping that the company will make more of this product!",
        "I salute the designers of this product.",
        "What's wrong with this company making the product?",
        "I regret buying this product. I hate it!"
    ]

    num_classes = len(default_labels)

    # Get user input for texts and their corresponding labels
    train_data = []
    for i in range(num_classes):
        class_label = st.text_input(f"Enter the label for class {i+1}:", value=default_labels[i])
        texts_input = st.text_area(f"Enter the texts for class {i+1} (one per line):", value="\n".join(default_texts[i]))

        if class_label and texts_input:  # Check if class label and texts have been entered
            texts = texts_input.splitlines()
            train_data.extend([(text, class_label) for text in texts])

    # Create a button to run the classification
    if st.button("Classify"):
        if train_data:  # Check if any training data has been entered
            # Convert data to a dataframe
            train_df = pd.DataFrame(train_data, columns=['Text', 'Label'])

            # Fit the classifier on the training data
            clf_few_shot.fit(train_df['Text'], train_df['Label'])

            # Get user input for texts to be classified
            user_input = st.text_area("Enter the texts to classify (one per line):", value="\n".join(default_input_to_classify))

            if user_input:  # Check if any text to be classified has been entered
                # Split the block of text into individual lines
                user_inputs = user_input.splitlines()

                # Predict the labels for the test data
                predicted_labels = clf_few_shot.predict(user_inputs)

                # Create a dataframe to display the results
                results_df = pd.DataFrame({
                    'Review': user_inputs,
                    'Predicted Label': predicted_labels
                })

                # Display the dataframe
                st.dataframe(results_df)
