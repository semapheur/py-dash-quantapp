# Tips
Clone this repository with 

```
git clone https://github.com/semapheur/py-dash-quantapp
```

## Virtual environment

### venv
Create a Python virtual environment with

```
python -m venv ./<name>
```

Activate the virtual environment using

```
# Windows
<name>\scripts\activate.ps1
```

### Conda

```
conda create --name <env> --file <this file>
```

## Libraries
Install required Python libraries with

```
pip install -r requirements.txt
```

## TailwindCSS
Install TailwindCSS with

```
# Node
npm install
```

Build the CSS file with

```
npm tw-build
```

## Application

Run the application with

```
python app.py
```

# XBRL taxonomy
https://xbrl.us/home/filers/sec-reporting/taxonomies/

# Jupyter

DataFrame to HTML table
```python
from IPython.display import display
```

# Accounting terms

## Revenue
- Revenues = 
    - RevenueFromContractWithCustomerIncludingAssessedTax + RevenueNotFromContractWithCustomer
    - InterestAndDividendIncomeOperating + RevenuesExcludingInterestAndDividends
    - InterestIncomeExpenseAfterProvisionForLoanLoss + NoninterestIncome
- SalesRevenueNet
- RevenuesExcludingInterestAndDividends =
  - BrokerageCommissionsRevenue + InvestmentBankingRevenue + UnderwritingIncomeLoss + PrincipalTransactionsRevenue + (FeesAndCommissions) + (InsuranceServicesRevenue) + MarketDataRevenue
- FinancialServicesRevenue = 
    - FeesAndCommissions + GainsLossesOnSalesOfAssets + PrincipalTransactionsRevenue + PremiumsEarnedNet + RevenueOtherFinancialServices
- InsuranceServicesRevenue = 
    - PremiumsEarnedNet + InsuranceInvestmentIncome + GainLossOnSaleOfInsuranceBlock + InsuranceAgencyManagementFee + InsuranceCommissionsAndFees

revenue_noninterest = dividend_income_operating + investment_banking_advisory_brokerage_underwriting_fees_commissions + revenue_principal_transaction + revenue_premiums + gain_loss_sale_financial_assets + gain_loss_sale_leased_assets_operating + gain_loss_sale_stock_subsidiary + gain_loss_sale_property_plant_equipment + gain_loss_sale_business + gain_loss_derivative_intstruments_pretax + revenue_noninterest_other + gain_loss_venture_capital + income_bank_owned_life_insurance + income_loss_real_estate_operation + revenue_real_estate_investment_partnership + gain_loss_conversion_investments_foreign + gain_loss_sale_stock_unissued + impairment_recovery_mortgage_servicing_rights + gain_loss_foreign_currency_transaction_pretax + gain_debt_conversion + gain_loss_extinguishment_debt + other_noninterest_operating_income

## Profit/Loss

- IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest = 
    - IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments - IncomeTaxExpenseBenefit + IncomeLossFromEquityMethodInvestments
    - IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest - IncomeTaxExpenseBenefit
- IncomeLossFromContinuingOperations = IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest - IncomeLossFromContinuingOperationsAttributableToNoncontrollingEntity
- IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest = IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments + IncomeLossFromEquityMethodInvestments
- IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments = IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic + IncomeLossFromContinuingOperationsBeforeIncomeTaxesForeign
- NetIncomeLossFromContinuingOperationsAvailableToCommonShareholdersBasic
- NetIncomeLossFromContinuingOperationsAvailableToCommonShareholdersDiluted
- IncomeLossBeforeExtraordinaryItemsAndCumulativeEffectOfChangeInAccountingPrinciple = IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest + IncomeLossFromDiscontinuedOperationsNetOfTax
- ProfitLoss = 
    - ExtraordinaryItemNetOfTax + IncomeLossBeforeExtraordinaryItemsAndCumulativeEffectOfChangeInAccountingPrinciple
    - NetIncomeLossIncludingPortionAttributableToNonredeemableNoncontrollingInterest - NetIncomeLossAttributableToRedeemableNoncontrollingInterest
    - IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest + IncomeLossFromDiscontinuedOperationsNetOfTax
- NetIncomeLoss = ProfitLoss - NetIncomeLossAttributableToNoncontrollingInterest
- NetIncomeLossAvailableToCommonStockholdersBasic = NetIncomeLoss - PreferredStockDividendsAndOtherAdjustments
- NetIncomeLossAvailableToCommonStockholdersDiluted = NetIncomeLossAvailableToCommonStockholdersBasic + InterestOnConvertibleDebtNetOfTax + ConvertiblePreferredDividendsNetOfTax + DilutiveSecurities

## Marketable securities
- Available for Sale (AFS) 
- Held to Maturity (HTM)
- Held for Trading (HFT)
- Fair Value through Net Income (FVNI)
- Fair Value through Other Comprehensive Income (FVOCI)