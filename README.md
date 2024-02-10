# Анализ на недвижими имоти спрямо описание в обява

## Идея на проекта

Този проект предлага дълбочинен анализ на данни за недвижими имоти, като се фокусира върху обработката и анализа на текстова информация в описанията на обявите. 
Основната цел е да се идентифицират шаблони, тенденции и потенциално взаимовръзки в данните, които могат да бъдат използвани за прогнозиране на пазарни тенденции, анализ на потребителското поведение или дори откриване на несъответствия и потенциално фалшиви обяви.
Тази информация е изключително ценна за инвеститори, агенти на недвижими имоти и купувачи, тъй като предоставя допълнителен слой анализ за по-информирано вземане на решения.

## Зависимости

Проектът използва няколко ключови Python библиотеки, които са основни за обработката на данни, машинното обучение, естествения език и визуализацията:

- `pandas`: За манипулация и анализ на данни.
- `matplotlib`: За създаване на визуализации.
- `numpy`: За поддръжка на високопроизводителни математически операции и многомерни масиви.
- `gensim`: За обучение на модели на естествен език.
- `sklearn`: За алгоритми на машинно обучение и кластеризация.
- `nltk`: За обработка на естествен език.
- `stop_words`: За филтриране на стоп думи от текстовете.
- `bulstem`: Специализирана библиотека за стеминг на български език.

## Инсталация

За да започнете работа с проекта, първо трябва да инсталирате всички необходими зависимости. Това може да стане лесно с помощта на следната команда:

```bash
pip install pandas matplotlib numpy gensim sklearn nltk stop_words bulstem
```

Уверете се, че имате инсталиран Python и pip на вашата система преди да изпълните командата.

## Употреба

След като всички зависимости са инсталирани, можете да стартирате скрипта, за да анализирате вашите данни. Процесът е разделен на няколко ключови стъпки:

1. **Поддава се файл за анализ**: Поддава се файл с обяви за анализ, файлът трябва да спазва следната структура(резултат след извикването на df.info(), където df е нашият pandas дейтафрейм за определен файл)
     ![image](https://github.com/MPTGits/RealEstateLookup/assets/37246713/f28305d3-d5bd-4a9f-9823-b5af6038a1fe)
3. **Предварителна обработка на текста**: Текстовите данни се обработват за премахване на стоп думи, пунктуация и ненужни символи.
4. **Обучение на Word2Vec модел**: Създава се модел за представяне на думите в многомерно пространство, за да може да се анализират техните взаимоотношения.
5. **Кластеризация на данните**: Използва се алгоритъмът DBSCAN за идентифициране на групи думи с подобно значение или употреба.
6. **Визуализация на резултатите**: Данните се визуализират, за да се покажат кластерите и да се анализират потенциалните шаблони.

### Визуализации

Примерни клъстери на обяви на недвижими имоти от Февруари 2023г. спрямо тяхното описание.

Долните резултати са в следствие на изпълнение на програмата със следните стойности:
- Word2Vec алгоритъм с дължина на векторите 100.
- TSNE с брой на компонентите 2 и смущение(preplexity) 5.
- DBSCAN алгоритъма с разстояние 7 и минимален брой на членове в клъстера 5.

![image](https://github.com/MPTGits/RealEstateLookup/assets/37246713/fb88ea89-95f3-483e-accf-51b9c5615619)

## Бъдещи подобрения

- Нужно е да се анализират допълнително потенциални стоп думи насочени конкретно за недвижими имоти и да се разшират ръчно добавените до този момент стоп думи към стандартните във файла stop_words.py
- Използване на още по-голям корпус от данни с обяви
- Да се премахват имена на агенции и изпълнители от описанието на обявите

## Ресурси
  - Източник на обяви: https://www.homes.bg/
  - STEM правила за български: https://github.com/mhardalov/bulstem-py/tree/master/bulstem/stemrules
