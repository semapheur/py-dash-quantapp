# Todo

- Implement sqlc

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


# Accounting terms

FloorBrokerageExchangeAndClearanceFees + MarketingAndAdvertisingExpense + DepreciationAndAmortization + AmortizationOfMortgageServicingRightsMSRs + RestructuringSettlementAndImpairmentProvisions + OtherNonrecurringIncomeExpense + OtherExpenses + RoyaltyExpense + AccretionExpenseIncludingAssetRetirementObligations + PreOpeningCosts + LegalFees - GainLossOnRepurchaseOfDebtInstrument + InducedConversionOfConvertibleDebtExpense + GeneralAndAdministrativeExpense - GainsLossesOnSalesOfAssets - GainLossOnSaleOfLeasedAssetsNetOperatingLeases - GainsLossesOnSalesOfOtherRealEstate - GainLossOnSaleOfStockInSubsidiaryOrEquityMethodInvestee - GainLossOnSaleOfPropertyPlantEquipment - GainLossOnSaleOfBusiness - GainsLossesOnExtinguishmentOfDebt - GainLossOnDerivativeInstrumentsNetPretax + BusinessCombinationIntegrationRelatedCosts

## Revenue

### GAAP

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

## Profit/Loss

### GAAP

- IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments =
    - OperatingIncomeLoss + NonoperatingIncomeExpense - (InterestAndDebtExpense)
    - IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic + IncomeLossFromContinuingOperationsBeforeIncomeTaxesForeign
- IncomeLossFromContinuingOperationsBeforeInterestExpenseInterestIncomeIncomeTaxesExtraordinaryItemsNoncontrollingInterestsNet = IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments + IncomeLossFromEquityMethodInvestments
- IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest = IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments + IncomeLossFromEquityMethodInvestments
- IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest = 
    - IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments - IncomeTaxExpenseBenefit + IncomeLossFromEquityMethodInvestments
    - IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest - IncomeTaxExpenseBenefit
- IncomeLossFromContinuingOperations = IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest - IncomeLossFromContinuingOperationsAttributableToNoncontrollingEntity
- NetIncomeLossFromContinuingOperationsAvailableToCommonShareholdersBasic
- NetIncomeLossFromContinuingOperationsAvailableToCommonShareholdersDiluted
- IncomeLossBeforeGainOrLossOnSaleOfPropertiesExtraordinaryItemsAndCumulativeEffectsOfAccountingChanges = IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest + IncomeLossFromDiscontinuedOperationsNetOfTax
- IncomeLossBeforeExtraordinaryItemsAndCumulativeEffectOfChangeInAccountingPrinciple = 
    - IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest + IncomeLossFromDiscontinuedOperationsNetOfTax
    - IncomeLossBeforeGainOrLossOnSaleOfPropertiesExtraordinaryItemsAndCumulativeEffectsOfAccountingChanges + GainLossOnSaleOfPropertiesNetOfApplicableIncomeTaxes
- IncomeLossIncludingPortionAttributableToNoncontrollingInterest = IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments + IncomeLossFromEquityMethodInvestments + DiscontinuedOperationIncomeLossFromDiscontinuedOperationBeforeIncomeTax + ExtraordinaryItemsGross
- ProfitLoss = 
    - ExtraordinaryItemNetOfTax + IncomeLossBeforeExtraordinaryItemsAndCumulativeEffectOfChangeInAccountingPrinciple
    - NetIncomeLossIncludingPortionAttributableToNonredeemableNoncontrollingInterest - NetIncomeLossAttributableToRedeemableNoncontrollingInterest + ExtraordinaryItemGainOrLossNetOfTaxAttributableToReportingEntity
    - IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest + IncomeLossFromDiscontinuedOperationsNetOfTax
    - InvestmentIncomeOperatingAfterExpenseAndTax + RealizedAndUnrealizedGainLossInvestmentAndDerivativeOperatingAfterTax + ForeignCurrencyTransactionGainLossAfterTax
- NetIncomeLoss = ProfitLoss - NetIncomeLossAttributableToNoncontrollingInterest
- NetIncomeLossAvailableToCommonStockholdersBasic = NetIncomeLoss - PreferredStockDividendsAndOtherAdjustments
- NetIncomeLossAvailableToCommonStockholdersDiluted = NetIncomeLossAvailableToCommonStockholdersBasic + InterestOnConvertibleDebtNetOfTax + ConvertiblePreferredDividendsNetOfTax + DilutiveSecurities

### IFRS
- ProfitLossFromContinuingOperations = ProfitLossBeforeTax - IncomeTaxExpenseContinuingOperations
- ProfitLoss = ProfitLossFromContinuingOperations + ProfitLossFromDiscontinuedOperations

### Gain/Loss

- revenue_finance_noninterest
    - revenue_services_financial + revenue_market_data + noninterest_income_operating_other + noninterest_income_other?
    - noninterest_income_operating + revenue_market_data + revenue_services_financial_other
- revenue_services_financial = [investment_banking_advisory_brokerage_underwriting_fees_commissions] + [gain_loss_sale_financial_assets] + [revenue_premiums] + [revenue_principal_transaction] + revenue_services_financial_other
- noninterest_income_operating = [investment_banking_advisory_brokerage_underwriting_fees_commissions] + [revenue_premiums] + [revenue_principal_transaction] + [gain_loss_sale_financial_assets] + noninterest_income_operating_other + noninterest_income_other
- gain_loss_sale_financial_assets = gain_loss_securitization_financial_assets + gain_loss_sale_securities + gain_loss_realized_securities + gain_loss_sale_loans + gain_loss_sale_mortgagebacked_securities + gain_loss_sale_insurance_block + [gain_loss_sale_derivatives] + gain_loss_sale_credit_card + gain_loss_sale_other_financial_assets
- gain_loss_realized_investments = gain_loss_securitization_financial_assets + gain_loss_realized_securities + gain_loss_sale_securities + gain_loss_sale_loans + gain_loss_sale_lease_financing + gain_loss_sale_leased_assets_operating + gain_loss_sale_properties_pretax + gain_loss_sale_investment_real_estate + gain_loss_sale_equity_method_investment + [gain_loss_sale_investment_other] + gain_loss_sale_other_financial_assets + gain_loss_sale_other_assets
- gain_loss_sale_investments = gain_loss_sale_stock_subsidiary_or_equity_method_investee + gain_loss_sale_investments_equity + gain_loss_sale_investments_debt + gain_loss_sale_properties + [gain_loss_sale_derivatives] + gain_loss_sale_commodity_contracts + [gain_loss_sale_investment_other]
- gain_loss_marketable_securities_cost_method_investments = gain_loss_realized_securities_marketable + gain_loss_cost_method_investments + gain_loss_realized_investments + gain_loss_sale_investments


## Cashflow

NetCashProvidedByUsedInOperatingActivitiesContinuingOperations = ProfitLoss - ExtraordinaryItemNetOfTax - IncomeLossFromDiscontinuedOperationsNetOfTax + AdjustmentsToReconcileNetIncomeLossToCashProvidedByUsedInOperatingActivities

ProfitLoss = 
    - ExtraordinaryItemNetOfTax + IncomeLossBeforeExtraordinaryItemsAndCumulativeEffectOfChangeInAccountingPrinciple
    - NetIncomeLossIncludingPortionAttributableToNonredeemableNoncontrollingInterest - NetIncomeLossAttributableToRedeemableNoncontrollingInterest + ExtraordinaryItemGainOrLossNetOfTaxAttributableToReportingEntity
    - IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest + IncomeLossFromDiscontinuedOperationsNetOfTax
    - InvestmentIncomeOperatingAfterExpenseAndTax + RealizedAndUnrealizedGainLossInvestmentAndDerivativeOperatingAfterTax + ForeignCurrencyTransactionGainLossAfterTax

## Asset valuation

- Carrying value/amount: book value
- Fair value: market value

Valuation adjustment:
- Accretion: unplanned increase in carrying value increase of a liability 
- Amortization: planned reduction in carrying value of intangible assets
- Depreciation: planned reducting in carrying value of tangible assets
- Impairment: unplanned reducting in carrying value of an asset (when fair value falls below carrying value)

Impairment of securities are classified as other than temporary (OTT) and are recognized either in earnings or comprehensive earnings.

### Marketable securities
- Available for Sale (AFS): fair value through other comprehensive income (FVOCI)
- Held to Maturity (HTM): carrying value at amortized cost basis
- Held for Trading (HFT): fair value through net income (FVNI)

# PDF/HTML table scrap training data

Registered period (duration/instant) reflects whether the financial date is reported over an interval or on an instant date.
