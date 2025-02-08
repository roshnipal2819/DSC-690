import { ChatOpenAI } from "@langchain/openai";
import * as dotenv from "dotenv";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createRetrievalChain } from "langchain/chains/retrieval";
import {HumanMessage} from "@langchain/core/messages";
import {AIMessage} from "@langchain/core/messages";
import {MessagesPlaceholder} from "@langchain/core/prompts";
import {createHistoryAwareRetriever} from "langchain/chains/history_aware_retriever";

dotenv.config();
process.noDeprecation = true;

const createVectorStore= async () => {
    const loader= new CheerioWebBaseLoader("https://js.langchain.com/docs/concepts/document_loaders");

    const docs = await loader.load();

    const splitter= new RecursiveCharacterTextSplitter({
        chunk_size: 200, chunk_overlap: 20, });

    const splitDocs=await splitter.splitDocuments(docs);

    const embeddings= new OpenAIEmbeddings();

    const vectorstore= await MemoryVectorStore.fromDocuments(
    splitDocs, embeddings, );

    return vectorstore;
};

const createChain= async () => {
    const model= new ChatOpenAI({
    modelName: "gpt-3.5-turbo", temperature: 0.7,});

    const prompt= ChatPromptTemplate.fromMessages([
        ["system", "Answer the following questions based on the following {context}."],
        new MessagesPlaceholder("chat_history"),
        ["user", "{input}"],
    ]);

    const chain= await createStuffDocumentsChain({
    llm: model, prompt: prompt, });

    const retriever = vectorstore.asRetriever({ k: 2 });

    const retrieverPrompt= ChatPromptTemplate.fromMessages([
       new MessagesPlaceholder("chat_history"),
       ["user", "{input}"],
       ["user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation"],
    ]);

    const historyAwareRetriever= await createHistoryAwareRetriever({
        llm: model, retriever, rephrasePrompt: retrieverPrompt, });

    const conversationChain=await createRetrievalChain({ combineDocsChain: chain, retriever, });
    return conversationChain;
};

const vectorstore= await createVectorStore();
const chain=await createChain(vectorstore);

const chatHistory= [
    new HumanMessage("Hello"),
    new AIMessage("Hi, how can i help you?"),
    new HumanMessage("My name is Leao."),
    new AIMessage("Hi leao, how can i help?"),
    new HumanMessage("what is LCEL?"),
    new AIMessage("LCEL stands for LangChain Expression Language."),
];
const response= await chain.invoke({
    input: "What is it?",
    chat_history: chatHistory,
});

console.log(response);
