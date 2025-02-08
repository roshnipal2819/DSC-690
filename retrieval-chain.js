import { ChatOpenAI } from "@langchain/openai";
import * as dotenv from "dotenv";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createRetrievalChain } from "langchain/chains/retrieval";

dotenv.config();
process.noDeprecation = true;

const model= new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    temperature: 0.7,
});

const prompt= ChatPromptTemplate.fromTemplate(`
            Answer the user's question. 
            Context: {context}
            Question: {input}  
`);

const chain= await createStuffDocumentsChain({
    llm: model,
    prompt: prompt,
});

// Load Data from WebPage
const loader= new CheerioWebBaseLoader("https://js.langchain.com/docs/introduction/");
const docs = await loader.load();

const splitter= new RecursiveCharacterTextSplitter({
    chunk_size: 200,
    chunk_overlap: 20,
    });

const splitDocs=await splitter.splitDocuments(docs);
// console.log(splitDocs);



const embeddings= new OpenAIEmbeddings();

const vectorstore= await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings,
    );

// Retrieval the Data
const retriever = vectorstore.asRetriever({ k: 2 });

const retrivalChain=await createRetrievalChain({
    combineDocsChain: chain,
    retriever,
});

const response= await retrivalChain.invoke({
    input: "What is LCEL?",
});

console.log(response);