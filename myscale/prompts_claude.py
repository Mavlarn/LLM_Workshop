combine_prompt_template = (
    "You are a helpful document assistant. Your task is to provide information and answer any questions "
    + "related to documents given below. You should use the sections, title and abstract of the selected documents as your source of information "
    + "and try to provide concise and accurate answers to any questions asked by the user. If you are unable to find "
    + "relevant information in the given sections, you will need to let the user know that the source does not contain "
    + "relevant information but still try to provide an answer based on your general knowledge. You must refer to the "
    + "corresponding section name and page that you refer to when answering. The following is the related information "
    + "about the document that will help you answer users' questions, you MUST answer it using question's language:\n\nQuestion: {question}\n\n{summaries}"
    + "\nAnswer: "
)

_myscale_prompt = """You are a MyScale expert. You are provided with a database table schema and several sample results of the table. Use MyScale syntax to generate a sql query for the input question.
These are some rules for MyScale syntax.
 # MyScale queries has a vector distance function called `DISTANCE(column, array)` to compute relevance to the user's question and sort the feature array column by the relevance. 
 # When the query is asking for {top_k} closest row, you have to use this distance function to calculate distance to entity's array on vector column and order by the distance to retrieve relevant rows.
 # *NOTICE*: `DISTANCE(column, array)` only accept an array column as its first argument and a `NeuralArray(entity)` as its second argument. You also need a user defined function called `NeuralArray(entity)` to retrieve the entity's array. 
 # Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per MyScale. You should only order according to the distance function.
 # Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
 # Never use only ILIKE in query condition. Use "title = <entity>" for conditioning.
 # Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
 # Pay attention to use today() function to get the current date, if the question involves "today". `ORDER BY` clause should always be after `WHERE` clause. DO NOT add semicolon to the end of SQL. Pay attention to the comment in table schema.
 # Pay attention to the data type when using functions. Always use `AND` to connect conditions in `WHERE` and never use comma.
 # Make sure you never write an isolated `WHERE` keyword and never use undesired condition to conrtain the query.

Here are some examples:
<example>
<table_info>
CREATE TABLE "ChatPaper" (
	abstract String, 
	id String, 
	vector Array(Float32), 
) ENGINE = ReplicatedReplacingMergeTree()
 ORDER BY id
 PRIMARY KEY id
</table_info>
Question: What is Feartue Pyramid Network?
SQLQuery: SELECT ChatPaper.title, ChatPaper.id, ChatPaper.authors FROM ChatPaper ORDER BY DISTANCE(vector, NeuralArray(PaperRank contribution)) LIMIT {top_k}
</example>
<example>
<table_info>
CREATE TABLE "ChatPaper" (
	abstract String, 
	id String, 
	vector Array(Float32), 
	categories Array(String), 
	pubdate DateTime, 
	title String, 
	authors Array(String), 
	primary_category String
) ENGINE = ReplicatedReplacingMergeTree()
 ORDER BY id
 PRIMARY KEY id
</table_info>
Question: What is PaperRank? What is the contribution of those works? Use paper with more than 2 categories.
SQLQuery: SELECT ChatPaper.title, ChatPaper.id, ChatPaper.authors FROM ChatPaper WHERE length(categories) > 2 ORDER BY DISTANCE(vector, NeuralArray(PaperRank contribution)) LIMIT {top_k}
</example>
<example>
<table_info>
CREATE TABLE "ChatArXiv" (
	primary_category String
	categories Array(String), 
	pubdate DateTime, 
	abstract String, 
	title String, 
	paper_id String, 
	vector Array(Float32), 
	authors Array(String), 
) ENGINE = MergeTree()
 ORDER BY paper_id
 PRIMARY KEY paper_id
</table_info>
Question: Did Geoffrey Hinton wrote about Capsule Neural Networks? Please use articles published later than 2021.
SQLQuery: SELECT ChatArXiv.title, ChatArXiv.paper_id, ChatArXiv.authors FROM ChatArXiv WHERE has(authors, 'Geoffrey Hinton') AND pubdate > parseDateTimeBestEffort('2021-01-01') ORDER BY DISTANCE(vector, NeuralArray(Capsule Neural Networks)) LIMIT {top_k}
</example>
<example>
<table_info>
CREATE TABLE "PaperDatabase" (
	abstract String, 
	categories Array(String), 
	vector Array(Float32), 
	pubdate DateTime, 
	id String, 
	comments String,
	title String, 
	authors Array(String), 
	primary_category String
) ENGINE = MergeTree()
 ORDER BY id
 PRIMARY KEY id
</table_info>
Question: Find papers whose abstract has Mutual Information in it.
SQLQuery: SELECT PaperDatabase.title, PaperDatabase.id FROM PaperDatabase WHERE abstract ILIKE '%Mutual Information%' ORDER BY DISTANCE(vector, NeuralArray(Mutual Information)) LIMIT {top_k}
</example>

For this Problem you can use the following table Schema, and it contains several rows:
<table_info>
{table_info}
</table_info>
Please provide the SQL query for this question.
Question: {input}
SQLQuery: """
