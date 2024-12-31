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

### Nonoperating income

- NonoperatingGainsLosses = GainLossOnInvestments + DisposalGroupNotDiscontinuedOperationGainLossOnDisposal + GainLossOnSaleOfStockInSubsidiaryOrEquityMethodInvestee + DeconsolidationGainOrLossAmount + GainLossOnSaleOfPreviouslyUnissuedStockBySubsidiaryOrEquityInvesteeNonoperatingIncome + GainLossOnSaleOfInterestInProjects + GainLossOnDerivativeInstrumentsNetPretax + BusinessCombinationBargainPurchaseGainRecognizedAmount + SaleLeasebackTransactionCurrentPeriodGainRecognized + OtherNonoperatingGainsLosses
- NonoperatingIncomeExpense = 
    - NonoperatingGainsLosses + InvestmentIncomeNet + RoyaltyIncomeNonoperating + RentalIncomeNonoperating + DevelopmentProfitsNonoperating + LeveragedLeasesIncomeStatementNetIncomeFromLeveragedLeases + ForeignCurrencyTransactionGainLossBeforeTax + OtherNonoperatingAssetRelatedIncome + OtherNonoperatingIncomeExpense
    - InvestmentIncomeNonoperating + GainLossOnContractTermination + GainLossOnCondemnation - LossFromCatastrophes + PublicUtilitiesAllowanceForFundsUsedDuringConstructionAdditions + ForeignCurrencyTransactionGainLossBeforeTax - SalesTypeLeaseInitialDirectCostExpenseCommencement - OperatingLeaseInitialDirectCostExpenseOverTerm + GainLossOnSaleOfLeasedAssetsNetOperatingLeases + GainsLossesOnSalesOfOtherRealEstate + BankOwnedLifeInsuranceIncome + RealEstateInvestmentPartnershipRevenue + ConversionGainsAndLossesOnForeignInvestments + ProfitLossFromRealEstateOperations - MortgageServicingRightsMSRImpairmentRecovery + DebtInstrumentConvertibleBeneficialConversionFeature + PublicUtilitiesAllowanceForFundsUsedDuringConstructionCapitalizedCostOfEquity - NetPeriodicDefinedBenefitsExpenseReversalOfExpenseExcludingServiceCostComponent + GovernmentAssistanceNonoperatingIncome + GovernmentAssistanceNonoperatingExpense + OtherNonoperatingIncomeExpense - UnusualOrInfrequentItemNetGainLoss


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
