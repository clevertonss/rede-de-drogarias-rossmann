# <font color='blue'> Predições de Vendas das Lojas ROSSMANN </font>

# Agenda

1. Contexto
2. Desafio
4. Desenvolvimento da Solução
5. Conclusão & Demonstração
6. Próximos Passos

# <font color='blue'> 1. Contexto </font>

- Reunião Mensal de Resultados
- CFO pediu uma Previsão de Vendas das Próximas 6 semanas de cada Loja

# <font color='blue'> 2. Desafio </font>

# Problema
- Definição do Budget para a Reforma das Lojas.

# Causas
- Predição de Vendas Atual apresentada muita Divergencia
- O processo de Predição de Vendas é baseado em Experiencias Passadas.
- Todo a Previsão de Vendas é feita Manualmente pelas 1.115 Lojas da Rossmann.
- A visualização das Vendas é Limitada ao Computador.

# Solução
- Usar Machine Learning para realizar a Previsão de Vendas de Todas as Lojas
- Visualização das Predições de Vendas poderão ser feitas pelo Smartphone

# <font color='blue'> 3. Desenvolvimento da Solução </font>

# DESCRICAO DOS DADOS


```python
print( 'Number of Rows: {}'.format( df1.shape[0] ) )
print( 'Number of Cols: {}'.format( df1.shape[1] ) )
```

    Number of Rows: 1017209
    Number of Cols: 18


# Descriptive Statistics


```python
# Central Tendency - mean, meadina 
ct1 = pd.DataFrame( num_attributes.apply( np.mean ) ).T
ct2 = pd.DataFrame( num_attributes.apply( np.median ) ).T

# dispersion - std, min, max, range, skew, kurtosis
d1 = pd.DataFrame( num_attributes.apply( np.std ) ).T 
d2 = pd.DataFrame( num_attributes.apply( min ) ).T 
d3 = pd.DataFrame( num_attributes.apply( max ) ).T 
d4 = pd.DataFrame( num_attributes.apply( lambda x: x.max() - x.min() ) ).T 
d5 = pd.DataFrame( num_attributes.apply( lambda x: x.skew() ) ).T 
d6 = pd.DataFrame( num_attributes.apply( lambda x: x.kurtosis() ) ).T 

# concatenar
m = pd.concat( [d2, d3, d4, ct1, ct2, d1, d5, d6] ).T.reset_index()
m.columns = ['attributes', 'min', 'max', 'range', 'mean', 'median', 'std', 'skew', 'kurtosis']
m
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>attributes</th>
      <th>min</th>
      <th>max</th>
      <th>range</th>
      <th>mean</th>
      <th>median</th>
      <th>std</th>
      <th>skew</th>
      <th>kurtosis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>store</td>
      <td>1.0</td>
      <td>1115.0</td>
      <td>1114.0</td>
      <td>558.429727</td>
      <td>558.0</td>
      <td>321.908493</td>
      <td>-0.000955</td>
      <td>-1.200524</td>
    </tr>
    <tr>
      <th>1</th>
      <td>day_of_week</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>6.0</td>
      <td>3.998341</td>
      <td>4.0</td>
      <td>1.997390</td>
      <td>0.001593</td>
      <td>-1.246873</td>
    </tr>
    <tr>
      <th>2</th>
      <td>sales</td>
      <td>0.0</td>
      <td>41551.0</td>
      <td>41551.0</td>
      <td>5773.818972</td>
      <td>5744.0</td>
      <td>3849.924283</td>
      <td>0.641460</td>
      <td>1.778375</td>
    </tr>
    <tr>
      <th>3</th>
      <td>customers</td>
      <td>0.0</td>
      <td>7388.0</td>
      <td>7388.0</td>
      <td>633.145946</td>
      <td>609.0</td>
      <td>464.411506</td>
      <td>1.598650</td>
      <td>7.091773</td>
    </tr>
    <tr>
      <th>4</th>
      <td>open</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.830107</td>
      <td>1.0</td>
      <td>0.375539</td>
      <td>-1.758045</td>
      <td>1.090723</td>
    </tr>
    <tr>
      <th>5</th>
      <td>promo</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.381515</td>
      <td>0.0</td>
      <td>0.485758</td>
      <td>0.487838</td>
      <td>-1.762018</td>
    </tr>
    <tr>
      <th>6</th>
      <td>school_holiday</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.178647</td>
      <td>0.0</td>
      <td>0.383056</td>
      <td>1.677842</td>
      <td>0.815154</td>
    </tr>
    <tr>
      <th>7</th>
      <td>competition_distance</td>
      <td>20.0</td>
      <td>200000.0</td>
      <td>199980.0</td>
      <td>5935.442677</td>
      <td>2330.0</td>
      <td>12547.646829</td>
      <td>10.242344</td>
      <td>147.789712</td>
    </tr>
    <tr>
      <th>8</th>
      <td>competition_open_since_month</td>
      <td>1.0</td>
      <td>12.0</td>
      <td>11.0</td>
      <td>6.786849</td>
      <td>7.0</td>
      <td>3.311085</td>
      <td>-0.042076</td>
      <td>-1.232607</td>
    </tr>
    <tr>
      <th>9</th>
      <td>competition_open_since_year</td>
      <td>1900.0</td>
      <td>2015.0</td>
      <td>115.0</td>
      <td>2010.324840</td>
      <td>2012.0</td>
      <td>5.515591</td>
      <td>-7.235657</td>
      <td>124.071304</td>
    </tr>
    <tr>
      <th>10</th>
      <td>promo2</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.500564</td>
      <td>1.0</td>
      <td>0.500000</td>
      <td>-0.002255</td>
      <td>-1.999999</td>
    </tr>
    <tr>
      <th>11</th>
      <td>promo2_since_week</td>
      <td>1.0</td>
      <td>52.0</td>
      <td>51.0</td>
      <td>23.619033</td>
      <td>22.0</td>
      <td>14.310057</td>
      <td>0.178723</td>
      <td>-1.184046</td>
    </tr>
    <tr>
      <th>12</th>
      <td>promo2_since_year</td>
      <td>2009.0</td>
      <td>2015.0</td>
      <td>6.0</td>
      <td>2012.793297</td>
      <td>2013.0</td>
      <td>1.662657</td>
      <td>-0.784436</td>
      <td>-0.210075</td>
    </tr>
    <tr>
      <th>13</th>
      <td>is_promo</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.155231</td>
      <td>0.0</td>
      <td>0.362124</td>
      <td>1.904152</td>
      <td>1.625796</td>
    </tr>
  </tbody>
</table>
</div>



# Mapa Mental de Hipoteses


![alt text](https://github.com/clevertonss/rede-de-drogarias-rossmann/blob/main/img/output_13_0.png)

# Hipoteses Da Análise Exploratória

**1.** Lojas com maior sortimentos deveriam vender mais.

**2.** Lojas com competidores mais próximos deveriam vender menos.

**3.** Lojas com competidores à mais tempo deveriam vendem mais.

**4.** Lojas com promoções ativas por mais tempo deveriam vender mais.

**5.** Lojas com mais dias de promoção deveriam vender mais.

**7.** Lojas com mais promoções consecutivas deveriam vender mais.

**8.** Lojas abertas durante o feriado de Natal deveriam vender mais.

**9.** Lojas deveriam vender mais ao longo dos anos.

**10.** Lojas deveriam vender mais no segundo semestre do ano.

**11.** Lojas deveriam vender mais depois do dia 10 de cada mês.

**12.** Lojas deveriam vender menos aos finais de semana.

**13.** Lojas deveriam vender menos durante os feriados escolares.


# ANALISE EXPLORATORIA DOS DADOS

# Response Variable


```python
sns.distplot( df4['sales'], kde=False  )
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11f7a3910>





![alt text](https://github.com/clevertonss/rede-de-drogarias-rossmann/blob/main/img/output_18_1.png)
    


# Numerical Variable


```python
num_attributes.hist( bins=25 );
```


![alt text](https://github.com/clevertonss/rede-de-drogarias-rossmann/blob/main/img/output_20_0.png)

    


# Categorical Variable


```python
# state_holiday
plt.subplot( 3, 2, 1 )
a = df4[df4['state_holiday'] != 'regular_day']
sns.countplot( a['state_holiday'] )

plt.subplot( 3, 2, 2 )
sns.kdeplot( df4[df4['state_holiday'] == 'public_holiday']['sales'], label='public_holiday', shade=True )
sns.kdeplot( df4[df4['state_holiday'] == 'easter_holiday']['sales'], label='easter_holiday', shade=True )
sns.kdeplot( df4[df4['state_holiday'] == 'christmas']['sales'], label='christmas', shade=True )

# store_type
plt.subplot( 3, 2, 3 )
sns.countplot( df4['store_type'] )

plt.subplot( 3, 2, 4 )
sns.kdeplot( df4[df4['store_type'] == 'a']['sales'], label='a', shade=True )
sns.kdeplot( df4[df4['store_type'] == 'b']['sales'], label='b', shade=True )
sns.kdeplot( df4[df4['store_type'] == 'c']['sales'], label='c', shade=True )
sns.kdeplot( df4[df4['store_type'] == 'd']['sales'], label='d', shade=True )

# assortment
plt.subplot( 3, 2, 5 )
sns.countplot( df4['assortment'] )

plt.subplot( 3, 2, 6 )
sns.kdeplot( df4[df4['assortment'] == 'extended']['sales'], label='extended', shade=True )
sns.kdeplot( df4[df4['assortment'] == 'basic']['sales'], label='basic', shade=True )
sns.kdeplot( df4[df4['assortment'] == 'extra']['sales'], label='extra', shade=True )
```




    <matplotlib.axes._subplots.AxesSubplot at 0x15bf1af40>




![alt text](https://github.com/clevertonss/rede-de-drogarias-rossmann/blob/main/img/output_22_1.png)
    


#  Validação das Hipóteses

### **H1.** Lojas com maior sortimentos deveriam vender mais.
**FALSA** Lojas com MAIOR SORTIMENTO vendem MENOS.


```python
aux1 = df4[['assortment', 'sales']].groupby( 'assortment' ).sum().reset_index()
sns.barplot( x='assortment', y='sales', data=aux1 );

aux2 = df4[['year_week', 'assortment', 'sales']].groupby( ['year_week','assortment'] ).sum().reset_index()
aux2.pivot( index='year_week', columns='assortment', values='sales' ).plot()

aux3 = aux2[aux2['assortment'] == 'extra']
aux3.pivot( index='year_week', columns='assortment', values='sales' ).plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x171f91a30>




![alt text](https://github.com/clevertonss/rede-de-drogarias-rossmann/blob/main/img/output_25_1.png)
    




![alt text](https://github.com/clevertonss/rede-de-drogarias-rossmann/blob/main/img/output_25_2.png)
  
    



![alt text](https://github.com/clevertonss/rede-de-drogarias-rossmann/blob/main/img/output_25_3.png)

    


### **H2.** Lojas com competidores mais próximos deveriam vender menos.
**FALSA** Lojas com COMPETIDORES MAIS PROXIMOS vendem MAIS.


```python
aux1 = df4[['competition_distance', 'sales']].groupby( 'competition_distance' ).sum().reset_index()

plt.subplot( 1, 3, 1 )
sns.scatterplot( x ='competition_distance', y='sales', data=aux1 );

plt.subplot( 1, 3, 2 )
bins = list( np.arange( 0, 20000, 1000) )
aux1['competition_distance_binned'] = pd.cut( aux1['competition_distance'], bins=bins )
aux2 = aux1[['competition_distance_binned', 'sales']].groupby( 'competition_distance_binned' ).sum().reset_index()
sns.barplot( x='competition_distance_binned', y='sales', data=aux2 );
plt.xticks( rotation=90 );

plt.subplot( 1, 3, 3 )
x = sns.heatmap( aux1.corr( method='pearson' ), annot=True );
bottom, top = x.get_ylim()
x.set_ylim( bottom+0.5, top-0.5 );
```


![alt text](https://github.com/clevertonss/rede-de-drogarias-rossmann/blob/main/img/output_27_0.png)

    


### **H8.** Lojas abertas durante o feriado de Natal deveriam vender mais.
**FALSA** Lojas abertas durante o feriado do Natal vendem menos.


```python
aux = df4[df4['state_holiday'] != 'regular_day']

plt.subplot( 1, 2, 1 )
aux1 = aux[['state_holiday', 'sales']].groupby( 'state_holiday' ).sum().reset_index()
sns.barplot( x='state_holiday', y='sales', data=aux1 );

plt.subplot( 1, 2, 2 )
aux2 = aux[['year', 'state_holiday', 'sales']].groupby( ['year', 'state_holiday'] ).sum().reset_index()
sns.barplot( x='year', y='sales', hue='state_holiday', data=aux2 );
```



![alt text](https://github.com/clevertonss/rede-de-drogarias-rossmann/blob/main/img/output_29_0.png)

    


### **H11.** Lojas deveriam vender mais depois do dia 10 de cada mês.
**VERDADEIRA** Lojas vendem mais depois do dia 10 de cada mes.


```python
aux1 = df4[['day', 'sales']].groupby( 'day' ).sum().reset_index()

plt.subplot( 2, 2, 1 )
sns.barplot( x='day', y='sales', data=aux1 );

plt.subplot( 2, 2, 2 )
sns.regplot( x='day', y='sales', data=aux1 );

plt.subplot( 2, 2, 3 )
sns.heatmap( aux1.corr( method='pearson' ), annot=True );

aux1['before_after'] = aux1['day'].apply( lambda x: 'before_10_days' if x <= 10 else 'after_10_days' )
aux2 =aux1[['before_after', 'sales']].groupby( 'before_after' ).sum().reset_index()

plt.subplot( 2, 2, 4 )
sns.barplot( x='before_after', y='sales', data=aux2 );
```



![alt text](https://github.com/clevertonss/rede-de-drogarias-rossmann/blob/main/img/output_31_0.png)
    


#  Resumo das Hipoteses


```python
tab =[['Hipoteses', 'Conclusao', 'Relevancia'],
      ['H1', 'Falsa', 'Baixa'],  
      ['H2', 'Falsa', 'Media'],  
      ['H3', 'Falsa', 'Media'],
      ['H4', 'Falsa', 'Baixa'],
      ['H5', '-', '-'],
      ['H7', 'Falsa', 'Baixa'],
      ['H8', 'Falsa', 'Media'],
      ['H9', 'Falsa', 'Alta'],
      ['H10', 'Falsa', 'Alta'],
      ['H11', 'Verdadeira', 'Alta'],
      ['H12', 'Verdadeira', 'Alta'],
      ['H13', 'Verdadeira', 'Baixa'],
     ]  
print( tabulate( tab, headers='firstrow' ) )
```

    Hipoteses    Conclusao    Relevancia
    -----------  -----------  ------------
    H1           Falsa        Baixa
    H2           Falsa        Media
    H3           Falsa        Media
    H4           Falsa        Baixa
    H5           -            -
    H7           Falsa        Baixa
    H8           Falsa        Media
    H9           Falsa        Alta
    H10          Falsa        Alta
    H11          Verdadeira   Alta
    H12          Verdadeira   Alta
    H13          Verdadeira   Baixa


# Analise Multivariada

# Numerical Attributes


```python
correlation = num_attributes.corr( method='pearson' )
sns.heatmap( correlation, annot=True );
```



![alt text](https://github.com/clevertonss/rede-de-drogarias-rossmann/blob/main/img/output_36_0.png)
    


# Categorical Attributes


```python
# only categorical data
a = df4.select_dtypes( include='object' )

# Calculate cramer V
a1 = cramer_v( a['state_holiday'], a['state_holiday'] )
a2 = cramer_v( a['state_holiday'], a['store_type'] )
a3 = cramer_v( a['state_holiday'], a['assortment'] )

a4 = cramer_v( a['store_type'], a['state_holiday'] )
a5 = cramer_v( a['store_type'], a['store_type'] )
a6 = cramer_v( a['store_type'], a['assortment'] )

a7 = cramer_v( a['assortment'], a['state_holiday'] )
a8 = cramer_v( a['assortment'], a['store_type'] )
a9 = cramer_v( a['assortment'], a['assortment'] )

# Final dataset
d = pd.DataFrame( {'state_holiday': [a1, a2, a3], 
               'store_type': [a4, a5, a6],
               'assortment': [a7, a8, a9]  })
d = d.set_index( d.columns )

sns.heatmap( d, annot=True )
```




    <matplotlib.axes._subplots.AxesSubplot at 0x122d1ad30>




![alt text](https://github.com/clevertonss/rede-de-drogarias-rossmann/blob/main/img/output_38_1.png)
  
    


# MACHINE LEARNING MODELLING

# Compare Model's Performance


```python
modelling_result_cv = pd.concat( [lr_result_cv, lrr_result_cv, rf_result_cv, xgb_result_cv] )
modelling_result_cv
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model Name</th>
      <th>MAE CV</th>
      <th>MAPE CV</th>
      <th>RMSE CV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Linear Regression</td>
      <td>2081.73 +/- 295.63</td>
      <td>0.3 +/- 0.02</td>
      <td>2952.52 +/- 468.37</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Lasso</td>
      <td>2116.38 +/- 341.5</td>
      <td>0.29 +/- 0.01</td>
      <td>3057.75 +/- 504.26</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Random Forest Regressor</td>
      <td>837.68 +/- 219.1</td>
      <td>0.12 +/- 0.02</td>
      <td>1256.08 +/- 320.36</td>
    </tr>
    <tr>
      <th>0</th>
      <td>XGBoost Regressor</td>
      <td>1030.28 +/- 167.19</td>
      <td>0.14 +/- 0.02</td>
      <td>1478.26 +/- 229.79</td>
    </tr>
  </tbody>
</table>
</div>



# <font color='blue'> 4. Conclusão & Demonstração </font>

# TRADUCAO E INTERPRETACAO DO ERRO

# Business Performance


```python
df92.sort_values( 'MAPE', ascending=False ).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store</th>
      <th>predictions</th>
      <th>worst_scenario</th>
      <th>best_scenario</th>
      <th>MAE</th>
      <th>MAPE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>291</th>
      <td>292</td>
      <td>104033.078125</td>
      <td>100714.973723</td>
      <td>107351.182527</td>
      <td>3318.104402</td>
      <td>0.565828</td>
    </tr>
    <tr>
      <th>908</th>
      <td>909</td>
      <td>238233.875000</td>
      <td>230573.337190</td>
      <td>245894.412810</td>
      <td>7660.537810</td>
      <td>0.520433</td>
    </tr>
    <tr>
      <th>875</th>
      <td>876</td>
      <td>203030.156250</td>
      <td>199110.952435</td>
      <td>206949.360065</td>
      <td>3919.203815</td>
      <td>0.305099</td>
    </tr>
    <tr>
      <th>721</th>
      <td>722</td>
      <td>353005.781250</td>
      <td>351013.625224</td>
      <td>354997.937276</td>
      <td>1992.156026</td>
      <td>0.268338</td>
    </tr>
    <tr>
      <th>594</th>
      <td>595</td>
      <td>400883.625000</td>
      <td>397415.263170</td>
      <td>404351.986830</td>
      <td>3468.361830</td>
      <td>0.242192</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.scatterplot( x='store', y='MAPE', data=df92 )
```




    <matplotlib.axes._subplots.AxesSubplot at 0x16a890280>




![alt text](https://github.com/clevertonss/rede-de-drogarias-rossmann/blob/main/img/output_45_1.png)    
    


# Total Performance


```python
df93 = df92[['predictions', 'worst_scenario', 'best_scenario']].apply( lambda x: np.sum( x ), axis=0 ).reset_index().rename( columns={'index': 'Scenario', 0:'Values'} )
df93['Values'] = df93['Values'].map( 'R${:,.2f}'.format )
df93
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Scenario</th>
      <th>Values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>predictions</td>
      <td>R$285,860,497.77</td>
    </tr>
    <tr>
      <th>1</th>
      <td>worst_scenario</td>
      <td>R$285,115,015.71</td>
    </tr>
    <tr>
      <th>2</th>
      <td>best_scenario</td>
      <td>R$286,605,979.84</td>
    </tr>
  </tbody>
</table>
</div>



# Machine Learning Performance


```python
plt.subplot( 2, 2, 1 )
sns.lineplot( x='date', y='sales', data=df9, label='SALES' )
sns.lineplot( x='date', y='predictions', data=df9, label='PREDICTIONS' )

plt.subplot( 2, 2, 2 )
sns.lineplot( x='date', y='error_rate', data=df9 )
plt.axhline( 1, linestyle='--')

plt.subplot( 2, 2, 3 )
sns.distplot( df9['error'] )

plt.subplot( 2, 2, 4 )
sns.scatterplot( df9['predictions'], df9['error'] )
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1689cf700>




![alt text](https://github.com/clevertonss/rede-de-drogarias-rossmann/blob/main/img/output_49_1.png)    
    


# <font color='blue'> 5. Próximos Passos </font>

- Workshop do Modelo para os Business Users
- Coletar Feedbacks sobre a Usabilidade
- Aumentar em 10% a Acurácia do Modelo

# <font color='blue'> Q & A </font>

# <font color='blue'> Muito Obrigado! </font>
