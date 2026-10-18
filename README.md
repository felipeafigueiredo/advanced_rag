# Advanced RAG — Família F2 (*How to Search*) sobre Normas Acadêmicas da UFRN

Projeto da Unidade 1 da disciplina **DCA0305 — Inteligência Artificial e Aprendizado de Máquina (UFRN)**.  
Avaliação empírica de técnicas de busca avançada da **Família F2 (*How to Search*)** comparadas ao pipeline RAG clássico (*baseline*) sobre um corpus de regulamentos e instruções normativas da UFRN.

---

## 1. Visão Geral e Problema (Família F2)

O RAG tradicional (cortes de tamanho fixo + busca vetorial densa única) apresenta três fragilidades estruturais na recuperação (*How to Search*):
1. **Cortes arbitrários (*Arbitrary cuts*):** Chunks de tamanho fixo separam o *caput* de um artigo dos seus parágrafos, incisos e exceções.
2. **Chunks isolados (*Isolated chunks*):** Informações distribuídas entre o Regulamento Geral e resoluções/instruções normativas específicas não são conectadas.
3. **Recuperador único (*One retriever*):** A busca puramente semântica (embeddings) falha ao recuperar termos exatos, siglas, números de artigos, prazos e pontuações de corte.

Para testar e mitigar essas falhas, avaliamos três técnicas da **Família F2** contra **3 rodadas do Baseline**:
* **Semantic Chunking:** Agrupa unidades normativas (artigos/parágrafos) e realiza o corte quando a similaridade de cosseno entre blocos adjacentes cai abaixo do limiar (`< 0,68`), gerando 209 chunks entre 650 e 1.350 caracteres.
* **Parent-Child Retrieval:** Indexa trechos filhos pequenos (~380 caracteres; 1.129 filhos) para busca densa de alta precisão, mas entrega ao gerador o trecho pai completo (1.500 caracteres) para preservar o contexto normativo.
* **Hybrid RAG (BM25 + RRF):** Combina a busca vetorial densa com a busca lexical **BM25** (sem *stopwords*) através do *Reciprocal Rank Fusion* ($k = 2$) sobre os 285 chunks do baseline.

---

## 2. Estrutura do Repositório

~~~text
.
├── README.md
├── corpus/                                  # 6 normas acadêmicas da UFRN (109 páginas convertidas para .md)
├── questions.json                           # 20 perguntas classificadas em 4 tipos com gabarito e fontes
├── notebook.ipynb                           # Pipeline completo executável de cima a baixo no Google Colab (GPU T4)
├── results/
│   ├── baseline_r1.csv                      # Rodada 1 do RAG tradicional
│   ├── baseline_r2.csv                      # Rodada 2 do RAG tradicional
│   ├── baseline_r3.csv                      # Rodada 3 do RAG tradicional
│   ├── results_semantic_chunking.csv        # Resultados: Semantic Chunking
│   ├── results_parent_child.csv             # Resultados: Parent-Child
│   ├── results_hybrid_rrf.csv               # Resultados: Hybrid (Dense + BM25 + RRF)
│   └── tabela_comparativa_final.csv         # Métricas consolidadas por experimento e tipo de pergunta
├── paper.pdf                                # Relatório técnico (2 páginas)
└── slides.pdf                               # Apresentação do projeto
~~~

---

## 3. Corpus e Conjunto de Perguntas (`questions.json`)

### Corpus (`corpus/`)
| ID | Documento | Assunto | Páginas |
| :--- | :--- | :--- | :---: |
| `regulamento_graduacao` | Resolução Nº 016/2023-CONSEPE | Regulamento dos Cursos de Graduação da UFRN | 82 |
| `res_013_2022` | Resolução Nº 013/2022-CONSEPE | Reocupação de vagas residuais | 9 |
| `in_08_2023` | IN Nº 8/2023-PROGRAD | Depósito do TCC de graduação no Repositório Institucional | 7 |
| `in_05_2022` | IN Nº 5/2022-PROGRAD | Solicitações de colação de grau individual | 4 |
| `in_10_2025` | IN Nº 10/2025-PROGRAD | Registro de diplomas digitais de outras IES do RN | 4 |
| `in_11_2026` | IN Nº 11/2026-PROGRAD | Revalidação de diploma estrangeiro (REVALIDA) | 3 |

### Categorias de Perguntas (*Tiers*)
As 20 perguntas foram desenhadas para expor cada ponto cego da família F2:
* **`simples` ($n = 5$):** Resposta contida em um único trecho (grupo de controle: Q5, Q6, Q9, Q10, Q11).
* **`termo-exato` ($n = 4$):** Exige recuperação literal de prazos, percentuais, notas de corte ou siglas (Q1, Q2, Q7, Q20).
* **`multi-chunk` ($n = 6$):** Resposta espalhada entre *caput*, parágrafos, incisos ou anexos do mesmo documento (Q3, Q4, Q8, Q12, Q13, Q14).
* **`entre-documentos` ($n = 5$):** Exige cruzar o Regulamento dos Cursos de Graduação com uma Resolução ou Instrução Normativa (Q15, Q16, Q17, Q18, Q19).

---

## 4. Como Reproduzir os Experimentos e a Tabela Final

O projeto foi construído para rodar integralmente no **Google Colab** com GPU gratuita (**T4**), sem necessidade de chaves de API externas:

1. Abra o arquivo `notebook.ipynb` no **Google Colab**.
2. No menu superior, acesse **Ambiente de execução -> Alterar o tipo de ambiente de execução** e selecione **T4 GPU**.
3. Clique em **Ambiente de execução -> Executar tudo**.
   * **Célula 1:** Instala o servidor local **Ollama** e baixa os modelos `nomic-embed-text` (embeddings) e `qwen2.5:3b` (gerador e avaliador automático).
   * **Célula 2:** Clona este repositório e carrega os 6 arquivos `.md` de `corpus/` e as 20 questões de `questions.json`.
   * **Célula 3:** Executa o *warm-up* da GPU, gera os embeddings das perguntas e define o gerador (`temperature=0.2`, `top_k=6`) e o avaliador automático (`acerto_auto`).
   * **Célula 4 e 5:** Constrói os índices do Baseline (chunks fixos de 1.000 caracteres, sobreposição de 200; 285 chunks) e das 3 técnicas da Família F2 (*Semantic Chunking*, *Parent-Child* e *BM25 + RRF*).
   * **Célula 6 e 7:** Executa as 3 rodadas do Baseline (`r1`, `r2`, `r3`) e os 3 experimentos F2, exporta os CSVs em `results/` e imprime a tabela comparativa consolidada.

> **Nota sobre Avaliação Manual (`acerto_manual`):** Além do juiz automático (`acerto_auto` via `qwen2.5:3b`), todas as 120 respostas foram avaliadas de forma cega por revisão humana (`acerto_manual`), adotada como referência principal devido à taxa de falsos positivos (31 em 120) do modelo-juiz de 3B parâmetros.

---

## 5. Resultados Consolidados

### Tabela Comparativa Principal (Referência: `acerto_manual` vs. `acerto_auto`)

| Experimento | Acurácia Manual (%) | Acurácia Auto (%) | Recall Doc (%) | Chamadas LLM / Q | Latência Média (s) | Termo-Exato (%) | Simples (%) | Entre-Docs (%) | Multi-Chunk (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline r1** | 50,0 | 75,0 | 100,0 | 1,0 | 2,43 | 50,0 | 60,0 | 60,0 | 33,3 |
| **Baseline r2** | 50,0 | 80,0 | 100,0 | 1,0 | 2,16 | 50,0 | 40,0 | 60,0 | 50,0 |
| **Baseline r3** | 45,0 | 60,0 | 100,0 | 1,0 | 2,29 | 25,0 | 40,0 | 60,0 | 50,0 |
| *Média Baseline* | *48,3* | *71,7* | *100,0* | *1,0* | *2,29* | *41,7* | *46,7* | *60,0* | *44,4* |
| **Semantic Chunking** | 40,0 | 55,0 | 100,0 | 1,0 | 3,38 | 25,0 | 40,0 | 40,0 | 50,0 |
| **Parent-Child** | 45,0 | 75,0 | 100,0 | 1,0 | 3,73 | 50,0 | 40,0 | 40,0 | 50,0 |
| **Hybrid RRF** | **55,0** | **85,0** | **100,0** | **1,0** | **2,88** | **75,0** | **40,0** | **60,0** | **50,0** |

*(Os percentuais por categoria nas colunas à direita referem-se à avaliação manual de referência).*

---

## 6. Principais Conclusões

1. **Melhor relação custo-benefício com Hybrid RRF:** Liderou tanto na avaliação manual (**55,0%**, +6,7 pp sobre a média do baseline) quanto na automática (**85,0%**, +13,3 pp), mantendo 1 chamada ao LLM por pergunta e acréscimo de apenas +0,6 s na latência média.
2. **Ganho focado em Termo Exato:** O *Hybrid RRF* elevou a acurácia em perguntas de termo exato para **75,0%** (contra 25% a 50% no baseline), recuperando trechos com números e critérios literais que a busca puramente densa deixou fora do top-6 (ex.: notas mínimas do ENEM no Art. 14 da Res. 013/2022 na Pergunta 7 e limites de 5% a 20% na Pergunta 2).
3. **Limitações de *Semantic Chunking* e *Parent-Child* com gerador 3B:** Embora o *recall* de documento tenha sido de 100% em todos os experimentos, chunks maiores entregues ao `qwen2.5:3b` aumentaram a latência (3,38 s e 3,73 s) e levaram o modelo pequeno a omitir partes de regras multi-critério.
4. **Necessidade de validação humana em modelos 3B:** O juiz automático (`qwen2.5:3b`) superestimou a acurácia em todos os experimentos (+15 pp a +30 pp), gerando 31 falsos positivos e 2 falsos negativos nas 120 respostas avaliadas.
