# Padrão didático dos notebooks

Cada conceito novo deve ser apresentado em camadas, nesta ordem:

1. **Problema:** qual pergunta o conceito resolve e por que o conceito anterior não basta.
2. **Intuição:** mecanismo em linguagem comum antes da notação.
3. **Vocabulário:** significado, unidade e momento de observação de cada símbolo.
4. **Hipóteses:** condições necessárias e o que falha quando elas não valem.
5. **Derivação curta:** de onde vem a fórmula, sem saltos algébricos essenciais.
6. **Implementação explícita:** correspondência entre linhas de código e quantidades matemáticas.
7. **Exemplo controlado:** parâmetros conhecidos e uma mudança por vez.
8. **Interpretação:** o que observar na tabela ou figura e o que não concluir.
9. **Armadilhas:** conceitos parecidos que não devem ser confundidos.
10. **Checagem:** exercício conceitual, gabarito e teste numérico.

## Perguntas de revisão

Antes de considerar uma seção concluída:

- um leitor consegue explicar o mecanismo sem repetir a fórmula?
- está claro quais quantidades são observadas, latentes, estimadas ou futuras?
- a notação foi definida antes de ser usada?
- há distinção entre propriedade populacional e resultado amostral?
- incerteza, horizonte e condicionamento estão explícitos?
- o gráfico possui uma leitura orientada?
- casos-limite e hipóteses frágeis foram mencionados?
- o código pode ser relacionado linha por linha à matemática?

## Profundidade esperada

O objetivo não é maximizar quantidade de álgebra. Aprofundamento significa tornar visíveis as
decisões e relações que permitem transferir o conceito para outro problema: mecanismo, condições,
interpretação, inferência e limites. Demonstrações formais longas podem ser extensões; passos que
mudam a interpretação não podem ser omitidos.

## Formulação matemática mínima

Intuição e visualização apoiam a matemática; não a substituem. Todo modelo apresentado deve
explicitar, quando aplicável:

1. o espaço de informação ou condicionamento, por exemplo $\mathcal F_t$;
2. a equação geradora do processo e o domínio de cada parâmetro;
3. as hipóteses sobre inovações, independência e existência de momentos;
4. esperança, variância, autocovariância e distribuição condicional relevantes;
5. condições de estacionariedade, causalidade, invertibilidade ou identificabilidade;
6. a derivação dos forecasts de média e variância, não apenas a fórmula final;
7. o estimador ou função objetivo usado, incluindo sua incerteza;
8. a correspondência entre símbolos matemáticos e variáveis no código;
9. casos-limite nos quais a expressão muda ou deixa de ser válida;
10. a distinção entre quantidade populacional, estimativa amostral e realização futura.

Cada notebook começa com uma **espinha matemática** que reúne a formulação central. As seções
seguintes derivam, simulam e testam cada parte dessa formulação para que o formulário não se torne
uma coleção de equações desconectadas.

